from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from ot import sliced_wasserstein_distance

_DEFAULT_SEASONS = {"Winter": [10, 11, 12, 1, 2, 3], "Summer": [4, 5, 6, 7, 8, 9]}


def build_selection_mask(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    model_col: str = "_model",
    year_col: str = "year",
) -> np.ndarray:
    """Return a boolean mask selecting rows whose (model, year) pair is in combos."""
    mi = pd.MultiIndex.from_arrays([df[model_col].to_numpy(), df[year_col].to_numpy()])
    return mi.isin(set(combos))


# %% Fast Wasserstein scoring functions (using pre-computed seasonal reference arrays and fixed projection matrix)


def build_season_references(
    df: pd.DataFrame,
    variables: Sequence[str],
    projections: np.ndarray,
    seasons: Optional[dict[str, list[int]]] = None,
) -> Dict[str, np.ndarray]:
    """Pre-project and pre-sort the reference data for each season.

    Intended to be called once before the main optimisation loop. By
    pre-projecting the fixed reference dataset onto the projection matrix
    and pre-sorting the result, the most expensive part of the SWD computation
    is paid exactly once per season rather than once per iteration.

    Parameters
    ----------
    df : pd.DataFrame
        Full reference dataset, must contain a 'Date' column.
    variables : Sequence[str]
        Variables used in the distance computation.
    projections : np.ndarray, shape (d, n_projections)
        Fixed projection matrix, e.g. from ot.sliced.get_random_projections.

    Returns
    -------
    Dict[str, np.ndarray]
        Per-season sorted projected reference arrays, shape (m, n_projections).
    """
    _seasons = _DEFAULT_SEASONS if seasons is None else seasons

    season_refs = {}
    for _season, months in _seasons.items():
        in_season = df["Date"].dt.month.isin(months).values
        full_data = df.loc[in_season, variables].to_numpy(dtype=np.float32)
        ref_proj = full_data @ projections  # (m, n_proj)
        season_refs[_season] = np.sort(ref_proj, axis=0)
    return season_refs


def build_references(
    df: pd.DataFrame,
    variables: Sequence[str],
    projections: np.ndarray,
) -> np.ndarray:
    """Pre-project and pre-sort the full reference data (non-seasonal).

    Analogous to build_season_references but for the full dataset.
    Intended to be called once before the main optimisation loop.

    Parameters
    ----------
    df : pd.DataFrame
        Full reference dataset.
    variables : Sequence[str]
        Variables used in the distance computation.
    projections : np.ndarray, shape (d, n_projections)
        Fixed projection matrix.

    Returns
    -------
    np.ndarray
        Sorted projected reference array, shape (m, n_projections).
    """
    full_data = df[variables].to_numpy(dtype=np.float32)
    ref_proj = full_data @ projections  # (m, n_proj)
    return np.sort(ref_proj, axis=0)


def _swd_precomputed(
    sub_data: np.ndarray,
    ref_proj_sorted: np.ndarray,
    projections: np.ndarray,
    p: int = 2,
) -> float:
    """Compute SWD against a pre-projected and pre-sorted reference.

    Same as (Python Optimal Transport) ot.sliced_wasserstein_distance, but for the case where the reference
    distribution is fixed across iterations. The reference projection and sort
    are pre-computed once in build_season_references. Only the candidate is
    projected and sorted here.

    The 1D Wasserstein distance between distributions of unequal size (n vs m)
    is computed by interpolating the candidate quantile function onto the
    reference quantile grid using np.interp, following the same approach as
    ot.lp.wasserstein_1d under uniform weights.

    Parameters
    ----------
    sub_data : np.ndarray, shape (n, d)
        Candidate selection, raw feature values.
    ref_proj_sorted : np.ndarray, shape (m, n_projections)
        Pre-sorted projected reference, from build_season_references.
    projections : np.ndarray, shape (d, n_projections)
        Fixed projection matrix.
    p : int
        Power parameter, consistent with sliced_wasserstein_distance.

    Returns
    -------
    float
        Sliced Wasserstein distance between candidate and reference.
    """
    m, n_proj = ref_proj_sorted.shape
    n = sub_data.shape[0]

    # Project and sort candidate — only work that varies per iteration
    cand_proj = sub_data.astype(np.float32) @ projections  # (n, n_proj)
    cand_proj_sorted = np.sort(cand_proj, axis=0)

    # Quantile grids for candidate and reference
    q_ref = np.linspace(0.0, 1.0, m, dtype=np.float32)  # (m,)
    q_cand = np.linspace(0.0, 1.0, n, dtype=np.float32)  # (n,)

    # Interpolate candidate onto reference quantile grid, one projection at a time
    cand_interp = np.column_stack(
        [np.interp(q_ref, q_cand, cand_proj_sorted[:, j]) for j in range(n_proj)]
    )  # (m, n_proj)

    # 1D Wasserstein per projection, aggregate to scalar
    projected_emd = np.mean(
        np.abs(cand_interp - ref_proj_sorted) ** p, axis=0
    )  # (n_proj,)
    return float((np.sum(projected_emd) / n_proj) ** (1.0 / p))


def wasserstein_seasonal_score(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    variables: Sequence[str],
    season_refs: Dict[str, np.ndarray],
    projections: np.ndarray,
    seasons: Optional[dict[str, list[int]]] = None,
    model_col: str = "model",
    year_col: str = "year",
    p: int = 2,
) -> float:
    """Wasserstein seasonal score using pre-computed reference projections.

    Replacement for POT-based wasserstein_seasonal_score intended for use in
    iterative optimisation loops. Requires
    season_refs and projections to be pre-computed once before the loop via
    build_season_references and ot.sliced.get_random_projections respectively.

    Parameters
    ----------
    df : pd.DataFrame
        Full reference dataset.
    combos : Sequence[Tuple[str, int]]
        Model-year combinations defining the candidate selection.
    variables : Sequence[str]
        Variables used in the distance computation.
    season_refs : Dict[str, np.ndarray]
        Pre-projected sorted reference arrays from build_season_references.
    projections : np.ndarray, shape (d, n_projections)
        Fixed projection matrix used to build season_refs.
    model_col : str
        Column name identifying the climate model.
    year_col : str
        Column name identifying the year.
    p : int
        Power parameter for the Wasserstein distance.

    Returns
    -------
    float
        SSWD score: maximum seasonal SWD across all seasons.
    """
    _seasons = _DEFAULT_SEASONS if seasons is None else seasons

    idx_sel = build_selection_mask(df, combos, model_col=model_col, year_col=year_col)
    wasserstein_seasonal = []
    for _season, months in _seasons.items():
        in_season = df["Date"].dt.month.isin(months).values
        sub_mask = idx_sel & in_season
        sub_data = df.loc[sub_mask, variables].to_numpy()
        wasserstein_seasonal.append(
            _swd_precomputed(sub_data, season_refs[_season], projections, p=p)
        )
    return max(wasserstein_seasonal)


def wasserstein_score(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    variables: Sequence[str],
    ref_proj_sorted: np.ndarray,
    projections: np.ndarray,
    model_col: str = "model",
    year_col: str = "year",
    p: int = 2,
) -> float:
    """Wasserstein score using pre-computed reference projections (non-seasonal).

    Replacement for POT wasserstein_score intended for use in iterative
    optimisation loops. Requires ref_proj_sorted and projections to be
    pre-computed once before the loop via build_references and
    ot.sliced.get_random_projections respectively.

    Parameters
    ----------
    df : pd.DataFrame
        Full reference dataset.
    combos : Sequence[Tuple[str, int]]
        Model-year combinations defining the candidate selection.
    variables : Sequence[str]
        Variables used in the distance computation.
    ref_proj_sorted : np.ndarray, shape (m, n_projections)
        Pre-sorted projected reference, from build_references.
    projections : np.ndarray, shape (d, n_projections)
        Fixed projection matrix used to build ref_proj_sorted.
    model_col : str
        Column name identifying the climate model.
    year_col : str
        Column name identifying the year.
    p : int
        Power parameter for the Wasserstein distance.

    Returns
    -------
    float
        SWD score between candidate and full reference.
    """
    idx_sel = build_selection_mask(df, combos, model_col=model_col, year_col=year_col)
    sub_data = df.loc[idx_sel, variables].to_numpy()
    return _swd_precomputed(sub_data, ref_proj_sorted, projections, p=p)


# %% Original POT-based scoring functions


def wasserstein_score_pot(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    variables: Sequence[str],
    model_col: str = "_model",
    year_col: str = "year",
    n_projections: int = 50,
    date_col: str = "Date",
    seasons: Optional[
        dict[str, list[int]]
    ] = None,  # unused; accepted for API consistency with seasonal variant
) -> float:
    """
    Sliced Wasserstein distance between the selected years and the full sample.

    Uses all rows across all seasons together.

    Parameters
    ----------
    df : daily DataFrame with model, year, and variable columns.
    combos : (model, year) pairs identifying the selected years.
    variables : variable columns to include in the distance calculation.
    n_projections : number of random projections. Higher = more accurate, slower.
                    50 is a reasonable default; 200 is used for precision boosts.
    date_col : unused here, kept for API consistency with the seasonal variant.
    seasons : unused here, kept for API consistency with the seasonal variant.

    Notes
    -----
    ``seed=42`` is passed to ``sliced_wasserstein_distance`` to fix the random
    projections. This ensures that two candidate selections evaluated within the
    same SA run always use identical projections, making scores directly
    comparable. The seed is intentionally hardcoded (not the SA seed) because
    the projection basis must be the same for every score call.
    """
    idx_sel = build_selection_mask(df, combos, model_col=model_col, year_col=year_col)
    sub_data = df.loc[idx_sel, list(variables)].to_numpy()
    full_data = df[list(variables)].to_numpy()
    return float(
        sliced_wasserstein_distance(
            sub_data, full_data, n_projections=n_projections, seed=42
        )
    )


def wasserstein_seasonal_score_pot(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    variables: Sequence[str],
    model_col: str = "_model",
    year_col: str = "year",
    n_projections: int = 50,
    date_col: str = "Date",
    seasons: Optional[dict[str, list[int]]] = None,
) -> float:
    """
    Seasonal sliced Wasserstein distance.

    Computes the sliced Wasserstein distance separately for each season defined
    in ``seasons``, then returns the maximum across all seasons. This ensures
    the selection is representative in every season rather than letting a good
    fit in one season mask a poor fit in another.

    Parameters
    ----------
    date_col : name of the datetime column in df, used for seasonal filtering.
    seasons : dict mapping season names to lists of month numbers (1–12).
              Defaults to ``{"Winter": [10, 11, 12, 1, 2, 3], "Summer": [4, 5, 6, 7, 8, 9]}``.
              Pass a custom dict to use different season boundaries, e.g. for
              the Southern Hemisphere or meteorological seasons.

    Notes
    -----
    ``seed=42`` is passed to ``sliced_wasserstein_distance`` to fix the random
    projections. This ensures that two candidate selections evaluated within the
    same SA run always use identical projections, making scores directly
    comparable. The seed is intentionally hardcoded (not the SA seed) because
    the projection basis must be the same for every score call.
    """
    _seasons = _DEFAULT_SEASONS if seasons is None else seasons

    idx_sel = build_selection_mask(df, combos, model_col=model_col, year_col=year_col)

    seasonal_scores = []
    for _season, months in _seasons.items():
        in_season = df[date_col].dt.month.isin(months).values
        sub_data = df.loc[idx_sel & in_season, list(variables)].to_numpy()
        full_data = df.loc[in_season, list(variables)].to_numpy()
        seasonal_scores.append(
            sliced_wasserstein_distance(
                sub_data, full_data, n_projections=n_projections, seed=42
            )
        )

    return float(max(seasonal_scores))
