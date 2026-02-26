# Divvy Bike-Share Demand Prediction

A geospatial machine-learning project that forecasts weekly bike-trip demand across Chicago's Divvy bike-share network. The model combines urban Points-of-Interest (POI) features with historical trip records, partitioned into 500 m × 500 m spatial grid cells, and trains a Random Forest regressor using a sliding-window time-series validation strategy.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Data Description](#data-description)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)

---

## Overview

The project is divided into two main components:

1. **Data Crawling & Preprocessing** — Fetches station metadata from the Divvy Bikes website and cleans, converts, and aggregates raw trip records by time period and rideable type.
2. **Demand Forecasting** — Merges historical trip counts with city-wide POI data on a 500 m grid, then trains and validates a `RandomForestRegressor` with a 52-week sliding window. The trained model is subsequently used to infer latent demand for grid cells that recorded zero trips.

---

## Repository Structure

```text
divvy/
├── data/
│   ├── chicago/                    # Chicago boundary Shapefile (for grid clipping)
│   ├── poi/                        # POI Shapefiles (retail, healthcare, education,
│   │                               #   sports, culture, parks, transit, metro,
│   │                               #   road network, land use, demographics)
│   ├── finished_trips/             # Weekly per-grid trip counts (CSV)
│   ├── aggregated_trip_counts/     # Trip counts aggregated by rideable type
│   │                               #   (electric / docked)
│   ├── dataset/                    # Pre-computed per-grid POI category counts
│   ├── road_weight/                # Road weight / capacity data
│   └── material/                   # Raw station data (JSON/XLSX) and trip samples
│                                   #   (CSV/XLSM)
├── src/
│   ├── process_data/               # Data preprocessing scripts
│   │   ├── add_station_coord.py    # Attach station coordinates to trip records
│   │   ├── convert_data.py         # Format conversion utilities
│   │   ├── divide_by_month.py      # Split trip data by month
│   │   ├── divide_by_week.py       # Split trip data by week
│   │   ├── divide_by_rideable_type.py  # Split trip data by bike type
│   │   ├── divvy_json2excel.py     # Convert station JSON to Excel
│   │   ├── export_grid.py          # Generate the geographic grid
│   │   ├── tripCountPerHour.py     # Aggregate trip counts per hour
│   │   ├── draw_CDF_chart.py       # Plot cumulative distribution function
│   │   ├── draw_PDF_chart.py       # Plot probability density function
│   │   └── ...                     # Additional helper scripts
│   └── prediction/                 # Forecasting pipeline scripts
│       ├── grid_trip.py            # Count trips per grid cell
│       ├── grid_poi.py             # Count POIs per grid cell
│       ├── aggregate_trip_count.py # Aggregate counts across multiple files
│       ├── training.py             # Model training (Random Forest + sliding window)
│       ├── reasoning.py            # Infer demand for zero-trip grid cells
│       └── ...                     # Additional helper scripts
├── output/                         # Generated outputs (git-ignored)
├── model/                          # Saved model files (git-ignored)
└── readme.md
```

---

## Data Description

| Directory | Contents |
|---|---|
| `chicago/` | Chicago administrative boundary Shapefile used for spatial grid clipping |
| `poi/` | POI Shapefiles across 10+ urban categories: retail, healthcare, education, sports, culture, parks, bus stops, metro stations, road network, land use, and demographic attributes |
| `finished_trips/` | Weekly per-grid trip counts; file naming convention: `trip_counts_week_YYYY-MM-DD.csv` |
| `aggregated_trip_counts/` | Trip count summaries split by rideable type (`electric`, `docked`) |
| `dataset/` | Per-grid POI category counts produced by `grid_poi.py` |
| `road_weight/` | Road weight and capacity data |
| `material/` | Raw station information (JSON/XLSX) and historical trip samples (CSV/XLSM) |

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Tabular data manipulation |
| `geopandas`, `shapely` | Geospatial operations and coordinate transforms |
| `scikit-learn` | `RandomForestRegressor`, `MinMaxScaler`, `train_test_split` |
| `joblib` | Model serialization |
| `tqdm` | Progress reporting |
| `openpyxl` | Excel file I/O |

Install all dependencies with:

```bash
pip install pandas numpy geopandas shapely scikit-learn joblib tqdm openpyxl
```

---

## Usage

Follow the four steps below in order.

### Step 1 — Data Preprocessing

```bash
# Split raw trip records into weekly files
python src/process_data/divide_by_week.py

# Attach station coordinates to trip records
python src/process_data/add_station_coord.py
```

### Step 2 — Spatial Feature Generation

```bash
# Count trips per 500 m grid cell (output: output/trip_count_week/)
python src/prediction/grid_trip.py

# Count POIs per 500 m grid cell (output: output/grid_poi_counts.csv)
python src/prediction/grid_poi.py

# Copy the generated file to the dataset directory
cp output/grid_poi_counts.csv data/dataset/grid_poi_counts.csv
```

### Step 3 — Model Training

```bash
python src/prediction/training.py
```

The training procedure:

- Iterates over the trip files using a **52-week sliding window** (step size: 1 week).
- **Features**: per-category POI counts, `month`, and `day_of_year`.
- **Target**: weekly trip count per grid cell.
- Retains the model with the lowest **NMAE** (Normalized Mean Absolute Error) across all windows.
- Saves the best model to `model/rfr_model.pkl`.

### Step 4 — Demand Inference

```bash
python src/prediction/reasoning.py
```

Loads the saved model and predicts latent trip demand for every grid cell that recorded zero trips. Results are written to `output/reasoning_result/`.

---

## Model Evaluation

| Metric | Description |
|---|---|
| **R²** (Coefficient of Determination) | Goodness of fit; higher is better (maximum 1.0) |
| **NMAE** (Normalized Mean Absolute Error) | MAE divided by the mean of the target values; lower is better |
