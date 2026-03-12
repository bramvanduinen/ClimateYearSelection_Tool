from typing import Callable, Optional, Sequence, Tuple, TypedDict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# %% TypedDicts for worker args / results


class _SAArgs(TypedDict):
    """Argument dict for SA worker function."""

    df: pd.DataFrame
    variables: list[str]
    n_select: int
    scoring_fn: Callable[..., float]
    model_col: str
    year_col: str
    temp_start: float
    temp_end: float
    cooling_rate: float
    max_iter: int
    precision_boost: bool
    seed: int


class SAResult(TypedDict):
    """Result dict returned by each SA experiment."""

    selected: list[Tuple[str, int]]
    score: float
    log_df: pd.DataFrame
    seed: int


# %% Core SA algorithm


def simulated_annealing_selection(
    df: pd.DataFrame,
    variables: Sequence[str],
    n_select: int,
    scoring_fn: Callable[..., float],
    model_col: str = "_model",
    year_col: str = "year",
    temp_start: float = 2.0,
    temp_end: float = 1e-3,
    cooling_rate: float = 0.9975,
    max_iter: int = 5_000,
    random_state: Optional[int] = None,
    precision_boost: bool = False,
    verbose: bool = True,
) -> Tuple[list[Tuple[str, int]], float, pd.DataFrame]:
    """
    Simulated annealing selection of representative years.

    Minimises `scoring_fn` over all possible subsets of n_select (model, year)
    combinations present in df.

    Parameters
    ----------
    df : daily DataFrame with model, year, and variable columns.
    variables : variable columns used by the scoring function.
    n_select : number of (model, year) combinations to select.
    scoring_fn : callable with signature
                 ``fn(df, combos, variables, model_col, year_col, **kwargs) -> float``.
                 Lower scores are better.
    temp_start : initial SA temperature.
    temp_end : stopping temperature.
    cooling_rate : multiplicative cooling factor per iteration.
    max_iter : hard cap on iterations (algorithm may stop earlier via temp_end).
    random_state : seed for the random number generator.
    precision_boost : if True, re-evaluate close candidates with n_projections=200
                      for higher accuracy. Requires scoring_fn to accept an
                      ``n_projections`` keyword argument.
    verbose : show a tqdm progress bar.

    Returns
    -------
    (best_combos, best_score, log_df)
    - best_combos : sorted list of (model, year) tuples
    - best_score  : float (lowest score found)
    - log_df      : DataFrame with columns
                    [iter, temp, score_tried, score_accepted, score_best, acc_rate]
    """
    rng = np.random.default_rng(random_state)

    # Reheat on stalls to escape plateaus
    REHEAT_G = 100 if cooling_rate < 0.99 else 300
    REHEAT_GAMMA = 1.8

    accepted = 0
    steps_since_best = 0

    all_combos: list[Tuple[str, int]] = list(
        df[[model_col, year_col]].drop_duplicates().itertuples(index=False, name=None)
    )

    idx = rng.choice(len(all_combos), n_select, replace=False)
    current: list[Tuple[str, int]] = [all_combos[i] for i in idx]
    best = current.copy()

    _score_kw = dict(df=df, variables=variables, model_col=model_col, year_col=year_col)
    current_score = best_score = scoring_fn(combos=current, **_score_kw)

    temp = temp_start
    log: list[dict] = [{"iter": 0, "temp": temp, "score_best": current_score}]

    n_it = int(min(max_iter, np.log(temp_end / temp_start) / np.log(cooling_rate)))
    iterator = (
        tqdm(range(1, max_iter + 1), total=n_it, desc="SA")
        if verbose
        else range(1, max_iter + 1)
    )

    for it in iterator:
        if temp < temp_end:
            break

        remaining = [c for c in all_combos if c not in current]

        # Mixed-move strategy: swap 1 year most of the time, occasionally more
        u = rng.random()
        if n_select < 10:
            k = 1 if u < 0.75 else int(rng.integers(2, 4))
        else:
            if u < 0.5:
                k = int(rng.integers(1, 4))
            elif u < 0.85:
                k = int(rng.integers(4, 7))
            else:
                k = int(rng.integers(7, 11))

        k = int(min(k, n_select, len(remaining)))
        out_idx = rng.choice(n_select, size=k, replace=False)
        in_idx = rng.choice(len(remaining), size=k, replace=False)

        neighbour = current.copy()
        for oi, ii in zip(np.atleast_1d(out_idx), np.atleast_1d(in_idx)):
            neighbour[oi] = remaining[int(ii)]

        neighbour_score = scoring_fn(combos=neighbour, **_score_kw)

        # Metropolis accept/reject — for close candidates optionally boost precision
        random_check = rng.random()
        delta_init = neighbour_score - current_score
        close = (neighbour_score / current_score < 1.1) or (
            random_check < np.exp(-delta_init / temp)
        )
        if close and precision_boost:
            neighbour_score = scoring_fn(
                combos=neighbour, n_projections=200, **_score_kw
            )

        delta = neighbour_score - current_score
        accept = (delta < 0) or (random_check < np.exp(-delta / temp))

        if accept:
            current, current_score = neighbour, neighbour_score
            accepted += 1
            if current_score < best_score:
                best, best_score = current.copy(), current_score
                steps_since_best = 0
        else:
            steps_since_best += 1

        log.append(
            {
                "iter": it,
                "temp": temp,
                "score_tried": neighbour_score,
                "score_accepted": current_score,
                "score_best": best_score,
                "acc_rate": accepted / it,
            }
        )

        # Reheat on stall to kick out of local plateaus
        if steps_since_best > REHEAT_G:
            temp *= REHEAT_GAMMA
            steps_since_best = 0

        temp *= cooling_rate

    return sorted(best), float(best_score), pd.DataFrame(log)


# %% Parallel runner


def _sa_worker(args: _SAArgs) -> SAResult:
    """
    Top-level picklable worker for use with ProcessPoolExecutor.

    Runs a single SA experiment and returns a result dict with keys:
    'selected', 'score', 'log_df', 'seed'.
    """
    selected, score, log_df = simulated_annealing_selection(
        df=args["df"],
        variables=args["variables"],
        n_select=args["n_select"],
        scoring_fn=args["scoring_fn"],
        model_col=args["model_col"],
        year_col=args["year_col"],
        temp_start=args["temp_start"],
        temp_end=args["temp_end"],
        cooling_rate=args["cooling_rate"],
        max_iter=args["max_iter"],
        random_state=args["seed"],
        precision_boost=args["precision_boost"],
        verbose=False,
    )
    return {
        "selected": selected,
        "score": score,
        "log_df": log_df,
        "seed": args["seed"],
    }


def run_sa_parallel(
    df: pd.DataFrame,
    variables: Sequence[str],
    n_select: int,
    scoring_fn: Callable[..., float],
    model_col: str,
    year_col: str,
    temp_start: float,
    temp_end: float,
    cooling_rate: float,
    max_iter: int,
    precision_boost: bool,
    n_experiments: int,
    base_seed: int,
    n_workers: Optional[int],
) -> list[SAResult]:
    """
    Run n_experiments SA instances in parallel with different random seeds.

    For n_experiments=1 the single run is executed in the current process with
    a visible progress bar. For n_experiments>1 a ProcessPoolExecutor is used
    and an outer progress bar tracks completed experiments.

    Returns
    -------
    List of result dicts sorted by score ascending (best first).
    Each dict contains: 'selected', 'score', 'log_df', 'seed'.

    Notes
    -----
    The DataFrame is pickled once per experiment submission. For very large
    DataFrames consider reducing n_experiments or pre-filtering variables.
    """
    seeds = [base_seed + i for i in range(n_experiments)]
    common: dict = dict(
        df=df,
        variables=variables,
        n_select=n_select,
        scoring_fn=scoring_fn,
        model_col=model_col,
        year_col=year_col,
        temp_start=temp_start,
        temp_end=temp_end,
        cooling_rate=cooling_rate,
        max_iter=max_iter,
        precision_boost=precision_boost,
    )
    args_list: list[_SAArgs] = [{**common, "seed": seed} for seed in seeds]

    if n_experiments == 1:
        # Single run: show inner SA progress bar directly (no subprocess overhead)
        selected, score, log_df = simulated_annealing_selection(
            df=df,
            variables=list(variables),
            n_select=n_select,
            scoring_fn=scoring_fn,
            model_col=model_col,
            year_col=year_col,
            temp_start=temp_start,
            temp_end=temp_end,
            cooling_rate=cooling_rate,
            max_iter=max_iter,
            random_state=seeds[0],
            precision_boost=precision_boost,
            verbose=True,
        )
        return [
            {"selected": selected, "score": score, "log_df": log_df, "seed": seeds[0]}
        ]

    print(f"Running {n_experiments} SA experiments in parallel...")
    results: list[SAResult] = []
    parallel = Parallel(
        n_jobs=n_workers if n_workers is not None else -1, return_as="generator"
    )
    with tqdm(total=len(args_list), desc="SA experiments") as pbar:
        for result in parallel(delayed(_sa_worker)(args) for args in args_list):
            results.append(result)
            pbar.update(1)

    return sorted(results, key=lambda x: x["score"])
