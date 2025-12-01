# Pollution Livability AI

An AI-powered spatial analysis platform for assessing urban livability based on pollution, traffic, vegetation, and other environmental factors.

## 🌍 Overview

This project creates a comprehensive livability scoring system that:
- Fetches and processes geospatial data (OSM, NDVI, population, elevation)
- Computes air quality and pollution metrics
- Builds ML models to predict livability scores
- Provides an interactive Streamlit dashboard for visualization and analysis

## 📁 Project Structure

```
pollution-livability-ai/
├── data/
│   ├── raw/           # Raw downloaded data
│   ├── interim/       # Cleaned intermediate data
│   ├── processed/     # Final feature store and outputs
│   └── models/        # Trained ML models
├── etl/               # Data extraction, transformation, loading
│   ├── utils/         # API clients, geo helpers
│   ├── fetch_osm.py   # OpenStreetMap data fetching
│   ├── fetch_ndvi.py  # Vegetation index processing
│   ├── fetch_population.py  # Population density
│   ├── fetch_elevation.py   # Elevation and flood risk
│   └── fetch_aq_async.py    # Air quality data (async)
├── features/          # Feature engineering
│   ├── make_grid.py   # Create spatial grid
│   ├── spatial_join.py      # Compute OSM features
│   ├── engineer_features.py # NDVI and raster features
│   ├── scoring.py     # Component scoring and livability
│   └── feature_config.yaml  # Feature weights config
├── models/            # ML model training and prediction
│   ├── train.py       # Spatial CV training
│   ├── predict.py     # Point prediction API
│   ├── evaluate.py    # Model evaluation
│   └── explain.py     # SHAP explainability
├── app/               # Streamlit web application
│   ├── streamlit_app.py     # Main app entry
│   ├── components/    # UI components
│   ├── layers/        # Map layers
│   ├── audio/         # TTS for accessibility
│   └── assets/        # CSS and static files
├── scripts/           # Automation scripts
│   └── run_all.sh     # Full pipeline runner
├── workflows/         # CI/CD workflows
│   └── daily_pipeline.yaml
├── tests/             # Test suite
└── requirements.txt   # Python dependencies
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Generate grid for a city (e.g., Kolkata)
python features/make_grid.py --bbox "22.45,88.25,22.65,88.45" --res 1000

# Fetch OSM data (roads, industrial areas)
python etl/fetch_osm.py --bbox "22.45,88.25,22.65,88.45"

# Compute spatial features
python features/spatial_join.py

# Run scoring pipeline
python -m features.scoring

# Train model (optional)
python models/train.py
```

### 3. Launch the App

```bash
streamlit run app/streamlit_app.py
```

## 📊 Features

### Data Sources
- **OpenStreetMap**: Roads, industrial areas, land use
- **Open-Meteo / OpenAQ**: Real-time air quality (PM2.5, PM10)
- **NDVI Rasters**: Vegetation index from satellite imagery
- **WorldPop**: Population density (or synthetic estimates)
- **SRTM/DEM**: Elevation and flood risk proxy

### Livability Components
| Component | Description | Weight |
|-----------|-------------|--------|
| Pollution | Air quality (PM2.5) | 35% |
| Traffic | Road density | 20% |
| Industry | Distance to industrial areas | 15% |
| Green | Vegetation (NDVI) | 15% |
| Population | Population density | 10% |
| Flood | Flood risk proxy | 5% |

### ML Model
- **Algorithm**: XGBoost with spatial cross-validation
- **Target**: Livability score (0-100)
- **Validation**: GroupKFold on spatial blocks
- **Explainability**: SHAP feature importance

## 🗺️ Streamlit App Features

- **Interactive Map**: Click to select locations
- **Heatmap Layers**: Visualize scores across the city
- **Single Location Analysis**: Detailed breakdown for any point
- **Two-Location Comparison**: Side-by-side comparison with recommendations
- **Audio Narration**: TTS accessibility using gTTS
- **Layer Controls**: Toggle pollution, traffic, NDVI, etc.

## 📋 API Usage

```python
from models.predict import predict_point

# Get livability score for a location
result = predict_point(lat=22.5726, lon=88.3639)
print(result['score'])  # e.g., 72.5
print(result['components'])  # breakdown by factor
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model.py -v
```

## 📦 Configuration

Edit `features/feature_config.yaml` to customize:

```yaml
grid:
  resolution_m: 1000  # Grid cell size

weights:
  pollution: 0.35
  traffic: 0.2
  industry: 0.15
  green: 0.15
  population: 0.1
```

## 🔄 Automation

The project includes GitHub Actions for daily data refresh:

```yaml
# .github/workflows/daily_pipeline.yaml
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
```

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
