from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Sequence

import pandas as pd

from climate_year_selection_tool.sa import run_sa_parallel
from climate_year_selection_tool.scores import (
    wasserstein_score,
    wasserstein_score_pot,
    wasserstein_seasonal_score,
    wasserstein_seasonal_score_pot,
)

_DUMMY_MODEL_COL = "_model"
_DUMMY_MODEL_VAL = "model"

_SCORING_FNS = {
    "wasserstein": wasserstein_score,
    "wasserstein_pot": wasserstein_score_pot,  # legacy, slower version
    "seasonal_wasserstein": wasserstein_seasonal_score,
    "seasonal_wasserstein_pot": wasserstein_seasonal_score_pot,  # legacy, slower version
}


# %% Result type


@dataclass
class SelectionResult:
    """
    Result of a `select_years` run.

    Attributes
    ----------
    selected : selected years. A list of ints when model_col=None (single-model
               mode), or a list of (model, year) tuples in multi-model mode.
    score : best score achieved across all experiments (lower is better).
    log_df : SA convergence log of the best run.
             Columns: iter, temp, score_tried, score_accepted, score_best, acc_rate.
    year_col : name of the year column used (e.g. ``'year'``, ``'hydro_year'``).
               Carried on the result so downstream code can label output correctly.
    all_runs : all experiment result dicts, sorted by score ascending.
               Each dict contains: 'selected', 'score', 'log_df', 'seed'.
    """

    selected: list
    score: float
    log_df: pd.DataFrame
    year_col: str = "year"
    all_runs: list = field(default_factory=list)


# %% Helper


def add_custom_year(
    df: pd.DataFrame,
    date_col: str = "Date",
    start_month: int = 1,
    year_col_name: str = "custom_year",
) -> pd.DataFrame:
    """
    Add a custom year column to df based on a shifted calendar.

    Each row is assigned the calendar year of its date after shifting back by
    ``start_month - 1`` months.  This means the custom year begins on the 1st
    of ``start_month`` every calendar year.

    Examples
    --------
    - start_month=1  → calendar year (no shift)
    - start_month=4  → hydrological year starting April 1
                        (Jan–Mar 2016 are assigned to year 2015)
    - start_month=10 → meteorological year starting October 1

    Parameters
    ----------
    df : DataFrame containing a datetime column named `date_col`.
    date_col : name of the datetime column.
    start_month : first month of the custom year (1–12).
    year_col_name : name to give the new column.

    Returns
    -------
    Copy of df with an additional integer column `year_col_name`.
    """
    df = df.copy()
    df[year_col_name] = (df[date_col] - pd.DateOffset(months=start_month - 1)).dt.year
    return df


# %% Main API


def select_years(
    df: pd.DataFrame,
    variables: Sequence[str],
    n_select: int,
    *,
    date_col: str = "Date",
    year_col: str = "year",
    model_col: Optional[str] = None,
    scoring: str = "seasonal_wasserstein",
    seasons: Optional[dict[str, list[int]]] = None,
    n_experiments: int = 10,
    temp_start: float = 2.0,
    temp_end: float = 1e-3,
    cooling_rate: float = 0.9975,
    max_iter: int = 5_000,
    n_workers: Optional[int] = None,
    random_state: int = 42,
    precision_boost: bool = True,
    n_projections: int = 50,
) -> SelectionResult:
    """
    Select representative years from a climate dataset using simulated annealing.

    The algorithm minimises the sliced Wasserstein distance between the
    distribution of the selected years and the full climatological distribution,
    ensuring good statistical representativeness across all provided variables.

    Input format
    ------------
    df should have one row per time step (e.g. daily) and contain at minimum:

    - a datetime column (``date_col``)
    - an integer year column (``year_col``) — use ``add_custom_year`` to create
      a hydrological or other custom year column first
    - one column per variable in ``variables``
    - optionally: a model/source column (``model_col``) for multi-model datasets

    Parameters
    ----------
    df : DataFrame with daily (or finer) rows.
    variables : column names of the climate/impact variables to match.
    n_select : number of representative years to select.
    date_col : name of the datetime column. Cast to datetime if needed.
    year_col : year identifier column.
    model_col : column identifying the model or source dataset. If None,
                all rows are treated as a single model (single-dataset mode)
                and ``selected`` in the result contains plain year integers.
    scoring : scoring metric to minimise. Options:

              ``"seasonal_wasserstein"`` (default)
                  Seasonal max of sliced Wasserstein distance (slow, POT-based).
                  Calculates the score separately for each season and takes the maximum, to
                  ensure good representativeness across seasons.

              ``"seasonal_wasserstein_fast"``
                  Same as above but uses pre-computed fixed projections for the reference.
                  Faster in iterative optimisation; recommended for most cases.

              ``"wasserstein"``
                  Sliced Wasserstein distance over the full annual distribution
                  (slow, POT-based).

              ``"wasserstein_fast"``
                  Same as above but uses pre-computed fixed projections for the reference.
                  Faster in iterative optimisation.

    seasons : custom season definitions for ``"seasonal_wasserstein"`` scoring.
              A dict mapping season names to lists of month numbers (1–12).
              Defaults to ``{"Winter": [10, 11, 12, 1, 2, 3], "Summer": [4, 5, 6, 7, 8, 9]}``.
              Example for Southern Hemisphere::

                  seasons={"Summer": [10, 11, 12, 1, 2, 3], "Winter": [4, 5, 6, 7, 8, 9]}

              Ignored when ``scoring="wasserstein"``.
    n_experiments : number of independent SA runs, each with a different seed.
                    The best result across all runs is returned in the main
                    result fields; all individual results are in ``.all_runs``.
    temp_start : initial SA temperature.
    temp_end : SA stopping temperature.
    cooling_rate : multiplicative cooling factor per iteration (e.g. 0.9975).
                   Lower values cool faster; typical range 0.99–0.9995.
    max_iter : hard cap on SA iterations per run. The algorithm may stop earlier
               once temp_end is reached.
    n_workers : number of parallel workers for experiments.
                Defaults to n_experiments (one worker per experiment).
    random_state : base seed. Experiment i uses seed ``random_state + i``.
    precision_boost : if True (default), close candidate solutions are
                      re-evaluated with n_projections=200 for higher accuracy.
                      Set False only when using a custom scoring function that
                      does not accept an ``n_projections`` keyword argument.
    n_projections : number of random 1-D projections used for the sliced
                    Wasserstein distance. Higher values increase accuracy but
                    also runtime. 50 is a good default; use 200 for
                    publication-quality results.

    Returns
    -------
    SelectionResult
        .selected   list of ints (single-model) or (model, year) tuples.
        .score      best score achieved (lower is better).
        .log_df     SA convergence log DataFrame of the best run.
        .all_runs   all experiment results sorted by score ascending.
    """
    if scoring not in _SCORING_FNS:
        raise ValueError(
            f"scoring must be one of {list(_SCORING_FNS)}, got '{scoring}'"
        )

    # Validate variables exist in the DataFrame
    missing_vars = [v for v in variables if v not in df.columns]
    if missing_vars:
        raise ValueError(
            f"Variables not found in DataFrame columns: {missing_vars}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    single_model = model_col is None
    if single_model:
        df[_DUMMY_MODEL_COL] = _DUMMY_MODEL_VAL
        effective_model_col = _DUMMY_MODEL_COL
    else:
        effective_model_col = model_col

    # Validate n_select does not exceed the number of available years per model
    years_per_model = df.groupby(effective_model_col)[year_col].nunique()
    min_years = int(years_per_model.min())
    if n_select > min_years:
        model_info = "" if single_model else " (fewest in any model)"
        raise ValueError(
            f"n_select ({n_select}) exceeds the number of available years "
            f"({min_years}{model_info}). Reduce n_select or provide more data."
        )

    # Bake date_col and seasons into the scoring function so SA does not need to know about them
    scoring_fn = partial(_SCORING_FNS[scoring], date_col=date_col, seasons=seasons)

    all_runs = run_sa_parallel(
        df=df,
        variables=list(variables),
        n_select=n_select,
        scoring_fn=scoring_fn,
        model_col=effective_model_col,
        year_col=year_col,
        temp_start=temp_start,
        temp_end=temp_end,
        cooling_rate=cooling_rate,
        max_iter=max_iter,
        precision_boost=precision_boost,
        n_experiments=n_experiments,
        base_seed=random_state,
        n_workers=n_workers,
        n_projections=n_projections,
    )

    best = all_runs[0]  # sorted ascending by score

    if single_model:
        selected = [year for (_model, year) in best["selected"]]
    else:
        selected = list(best["selected"])

    return SelectionResult(
        selected=selected,
        score=best["score"],
        log_df=best["log_df"],
        year_col=year_col,
        all_runs=all_runs,
    )
