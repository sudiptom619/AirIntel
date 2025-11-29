#!/usr/bin/env bash
set -euo pipefail
# 1. fetch raw data
python etl/fetch_osm.py --bbox-file data/processed/city_grid.geojson
python -c "from etl.fetch_aq_async import run_fetch_pollution; run_fetch_pollution('data/processed/city_grid.geojson')"
python etl/fetch_ndvi.py
python etl/fetch_population.py
python etl/fetch_elevation.py

# 2. build features
python features/engineer_features.py  # ensure it calls integrate_features()

# 3. train model (optional)
python models/train.py

# 4. move model to data/models
