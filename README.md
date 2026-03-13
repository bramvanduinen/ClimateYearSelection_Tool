# Climate Year Selection Tool



Select a small subset of **representative climate years** from a larger climate dataset for usage in a downstream (e.g. impact) model or analysis that is too computationally expensive to run on the full dataset. The selected years collectively match the statistical distribution of the full dataset as closely as possible, across all your chosen variables and potentially in multiple seasons.

> **Note:** This project is a work in progress.

## How it works

**Climate year selection** is the problem of finding a small subset of years from a larger climate dataset that is as statistically representative as possible of the full dataset. The subset should preserve not just average conditions but also variability, extremes, and correlations between variables. Climate year selection is required whenever a downstream (e.g. impact) model or analysis is too computationally expensive to run on the full dataset.

Finding the optimal subset is a combinatorial problem with [combinatorial explosion](https://en.wikipedia.org/wiki/Combinatorial_explosion). Exhaustive search is infeasible for any realistic dataset size, so a smart search strategy is needed.

This tool uses **[simulated annealing (SA)](https://towardsdatascience.com/an-introduction-to-a-powerful-optimization-technique-simulated-annealing-87fd1e3676dd/)**, a stochastic optimisation algorithm that starts from a random subset and iteratively proposes small changes (swapping years in and out). Improvements are always accepted; slightly worse solutions are accepted with a decreasing probability as the search progresses. This allows the algorithm to escape local minima early on while converging reliably over time. Multiple independent runs are launched in parallel and the best result is returned.

Each candidate subset is scored using the (sliced) **[ Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric)**, an optimal-transport metric that measures how well the subset reproduces the full multivariate distribution of the reference dataset. Lower score = more representative selection.

This tool is based on [van Duinen et al. (in prep.)]() — *Selecting representative climate years for (national to continental-scale) energy system studies* — which compares several climate year selection methods for application in energy system studies and finds simulated annealing superior across subset sizes and spatial domains. This tool is generalized to work for any application.

## Installation

The package is not yet on PyPI. Clone the repository and install using one of these options:

```bash
# Option A (recommended): install in editable mode, makes the package importable everywhere
# Uses pyprojec.toml
pip install -e .

# Option B: install only the dependencies, then add src/ to your Python path manually
pip install -r requirements.txt
export PYTHONPATH=/path/to/climate_year_selection_tool/src
```

## Quick start

```python
import pandas as pd
from climate_year_selection_tool import select_years

# Load your data — one row per timestep (e.g. day), one column per variable
df = pd.read_csv("data/your_data.csv", parse_dates=["Date"])
df["year"] = df["Date"].dt.year

result = select_years(
    df,
    variables=["temp", "wind", "solar"],  # columns to match
    n_select=5,                            # how many years to pick
)

print(result.selected)  # e.g. [1923, 1947, 1961, 1978, 1994]
print(f"Score: {result.score:.6f}")       # lower is better
```

## Input data format

Your DataFrame must have:

| Column | Type | Description |
|---|---|---|
| `Date` (or custom name) | `datetime64` | One row per time step (daily or finer) |
| `year` (or custom name) | `int` | Year identifier for each row |
| variable columns | `float` | The climate/impact variables to match |

The column names are flexible — pass `date_col=` and `year_col=` to `select_years()` if yours differ from the defaults.

## Scoring options

### `"seasonal_wasserstein"` (default, recommended)

Computes the sliced-Wasserstein distance separately for each season (default: Winter (Oct–Mar) and Summer (Apr–Sep)) and returns the maximum of the two. This ensures the selected years are representative in both seasons, not just on an annual average. Seasons can be customized.

### `"wasserstein"`

Computes the Wasserstein distance over the full annual distribution. Slightly faster, but may select years that are good on average while being unrepresentative in one season.

```python
result = select_years(df, variables=["temp"], n_select=5, scoring="wasserstein")
```

## Custom year definitions

By default `year` means the calendar year. If you need a hydrological year (e.g. starting April 1) or any other shifted calendar, use `add_custom_year()`:

```python
from climate_year_selection_tool import add_custom_year

# April-start hydrological year: Jan–Mar 2016 are labelled year 2015
df = add_custom_year(df, date_col="Date", start_month=4, year_col_name="hydro_year")

result = select_years(df, variables=["temp", "precip"], n_select=5, year_col="hydro_year")
```

Other examples:
- `start_month=10` → custom year starting October 1
- `start_month=1` → calendar year (no shift, same as default)

## Multi-model datasets

If your dataset contains data from multiple climate models or sources, add a model column and pass `model_col=`:

```python
# Stack your per-model DataFrames and add a label
df_multi = pd.concat([
    df_model_a.assign(model="model_a"),
    df_model_b.assign(model="model_b"),
], ignore_index=True)

result = select_years(
    df_multi,
    variables=["temp", "wind"],
    n_select=5,
    model_col="model",   # tell the tool which column identifies the model
)

# selected is now a list of (model, year) tuples
print(result.selected)  # e.g. [("model_a", 1952), ("model_b", 1978), ...]
```

The algorithm selects years across all models simultaneously, ensuring the combined selection is representative of the full multi-model distribution.

## Tuning the algorithm

The defaults work well for most datasets. The most important parameters to adjust are:

| Parameter | Default | Effect |
|---|---|---|
| `n_select` | — | Number of representative years to pick. |
| `n_experiments` | `10` | Independent SA runs. More runs → more reliable result, proportionally slower. |
| `cooling_rate` | `0.9975` | How fast the annealing cools. Typical range `0.99–0.9995`. Lower = faster but less thorough. |
| `max_iter` | `5000` | Hard cap on iterations per run. |
| `random_state` | `42` | Base seed for reproducibility. |
| `precision_boost` | `True` | Re-evaluate close candidates with higher accuracy. Disable only when using a custom scoring function that doesn't accept `n_projections`. |

For a quick exploratory run, use `n_experiments=3, cooling_rate=0.9, max_iter=2000`. For production use, keep the defaults or increase `n_experiments`.

## Inspecting results

```python
# All selected years (best run)
print(result.selected)

# Best score achieved (lower = more representative)
print(result.score)

# SA convergence log of the best run
print(result.log_df.columns.tolist())
# ['iter', 'temp', 'score_tried', 'score_accepted', 'score_best', 'acc_rate']

# All individual experiment results, sorted best-first
for run in result.all_runs:
    print(run["seed"], run["score"], run["selected"])
```

## Running the examples

```bash
# (optional) Generate synthetic data. Also already provided in data/synthetic_climate_data.csv
python examples/make_synthetic_data.py
```

Then open the interactive example notebook:

```
examples/example_usage.ipynb
```

It walks through several use cases step by step: basic year selection, hydrological years, multi-model datasets, and inspecting results.

## Project structure

```
climate_year_selection_tool/
├── src/climate_year_selection_tool/
│   ├── __init__.py      # public API
│   ├── selector.py      # select_years(), add_custom_year(), SelectionResult
│   ├── sa.py            # simulated annealing engine
│   ├── scores.py        # Wasserstein scoring functions
│   └── io.py            # save_result() utilities
├── examples/
│   ├── make_synthetic_data.py   # generate test data
│   └── example_usage.ipynb   # notebook with example for simple use-cases
├── experiments/         # perform your experiments here
├── data/                # put your input data here
├── results/             # experiment results
├── pyproject.toml
└── requirements.txt
```

## Dependencies

| Package | Purpose |
|---|---|
| `pandas` | DataFrame handling |
| `numpy` | Numerical operations |
| `POT` | Sliced Wasserstein distance (Python Optimal Transport) |
| `joblib` | Parallel SA experiments |
| `tqdm` | Progress bar |
| `pyarrow` | Parquet I/O support |
| `fastparquet` | Parquet I/O support |

## Contact
Developed by: Bram van Duinen  
Contact: bram.van.duinen@knmi.nl