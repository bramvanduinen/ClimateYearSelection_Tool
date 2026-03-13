# Climate Year Selection Tool

Select a small set of **representative years** from a climate dataset. The selected years collectively match the statistical distribution of the full dataset as closely as possible, across all your chosen variables and potentially in multiple seasons.

> **Note:** This project is a work in progress.


## How it works

The tool uses **simulated annealing** to search for the best subset of years. Each candidate subset is scored using the **sliced Wasserstein distance**, an optimal-transport metric that measures how well the joint-distribution of the selected years matches the full climatological joint-distribution of the reference. Lower score = more representative selection.

The search runs multiple independent experiments in parallel (with different random seeds) and returns the best result found.

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
# Generate synthetic data first
python examples/make_synthetic_data.py

# Run all five example use cases
python examples/all_example_usages.py

# Or run the quick single-experiment example
python examples/fast_example_usage.py
```

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
│   ├── all_example_usages.py   # runnable examples
│   └── fast_example_usage.py   # quick single-experiment example
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