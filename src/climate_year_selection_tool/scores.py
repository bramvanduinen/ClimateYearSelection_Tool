from typing import Optional, Sequence, Tuple

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


def wasserstein_score(
    df: pd.DataFrame,
    combos: Sequence[Tuple[str, int]],
    variables: Sequence[str],
    model_col: str = "_model",
    year_col: str = "year",
    n_projections: int = 50,
    date_col: str = "Date",
    seasons: Optional[dict] = None,  # unused; accepted for API consistency with seasonal variant
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
        sliced_wasserstein_distance(sub_data, full_data, n_projections=n_projections, seed=42)
    )


def wasserstein_seasonal_score(
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
