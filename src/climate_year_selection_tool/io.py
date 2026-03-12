"""
I/O utilities for persisting SelectionResult objects.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from climate_year_selection_tool.selector import SelectionResult


def _result_slug(exp_name: str, settings: dict) -> str:
    """Build a short filename stem encoding the key run settings."""
    year_col = settings.get("year_col", "year")
    parts = [
        exp_name,
        f"n{settings['n_select']}",
        settings["scoring"],
        f"nexp{settings['n_experiments']}",
        f"cr{settings['cooling_rate']}",
        f"seed{settings['random_state']}",
    ]
    if year_col != "year":
        parts.append(f"yc{year_col}")
    return "__".join(str(p) for p in parts)


def save_result(
    result: SelectionResult,
    exp_name: str,
    settings: dict,
    out_dir: Path | str,
) -> None:
    """Save a SelectionResult to *out_dir* using three files per experiment.

    Parameters
    ----------
    result : SelectionResult returned by ``select_years()``.
    exp_name : short label for this experiment (used in filenames).
    settings : dict of run settings to embed in the summary JSON.
               Should include at minimum: n_select, scoring, n_experiments,
               cooling_rate, random_state.
    out_dir : directory to write files into. Created if it does not exist.

    Files written
    -------------
    <slug>_summary.json  – selected years, best score, and all settings.
    <slug>_log.csv       – SA convergence log of the best run.
    <slug>_runs.csv      – score and selected years for every experiment run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = _result_slug(exp_name, settings)

    # Summary JSON
    summary = {
        "selected": result.selected,
        "score": result.score,
        "settings": settings,
    }
    (out_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    # SA convergence log of the best run
    result.log_df.to_csv(out_dir / f"{stem}_log.csv", index=False)

    # All runs summary
    runs_df = pd.DataFrame(
        [
            {"seed": r["seed"], "score": r["score"], "selected": str(r["selected"])}
            for r in result.all_runs
        ]
    )
    runs_df.to_csv(out_dir / f"{stem}_runs.csv", index=False)

    print(f"  → saved to {out_dir}/{stem}_*")
