"""Evaluate saved livability model and produce simple error maps and metrics.

Usage: python models/evaluate.py
Produces: data/models/predictions.geojson, data/models/eval_metrics.json, and PNG maps
in data/models/ (predicted, residual).
"""
import json
from pathlib import Path
import math

import joblib
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
FEATURE_STORE = ROOT / "data" / "processed" / "features.parquet"
MODEL_PATH = ROOT / "data" / "models" / "livability_model.pkl"
OUT_DIR = ROOT / "data" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "livability_score"
PREFERRED_FEATURES = [
    "pm25",
    "pm10",
    "road_density",
    "dist_to_industry",
    "ndvi_mean",
    "pop_density",
    "elev_mean",
    "flood_proxy",
]


def load_model(path=MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def load_features(path=FEATURE_STORE):
    if not Path(path).exists():
        raise FileNotFoundError(f"Feature store not found: {path}")
    try:
        gdf = gpd.read_parquet(path)
    except Exception:
        gdf = gpd.read_file(path)
    return gdf


def evaluate():
    print("Loading model and features...")
    model = load_model()
    gdf = load_features()

    # Determine which features the model expects or fall back to preferred/intersecting features
    feature_cols = None
    try:
        if hasattr(model, 'feature_names_in_'):
            feature_cols = list(model.feature_names_in_)
    except Exception:
        feature_cols = None

    if feature_cols is None:
        # choose intersection of preferred features and available columns
        feature_cols = [c for c in PREFERRED_FEATURES if c in gdf.columns]

    if len(feature_cols) == 0:
        raise RuntimeError("No usable feature columns found for prediction")

    print('Using feature columns for evaluation:', feature_cols)
    X = gdf[feature_cols].fillna(0).astype(float)

    print(f"Predicting {len(X)} rows")
    try:
        preds = model.predict(X)
    except Exception as e:
        # If model expects different input (column ordering), try passing numpy array
        preds = model.predict(X.values)

    gdf['predicted'] = preds
    if TARGET_COL in gdf.columns:
        gdf['residual'] = gdf['predicted'] - pd.to_numeric(gdf[TARGET_COL], errors='coerce')
    else:
        gdf['residual'] = pd.NA

    # Metrics
    metrics = {}
    if TARGET_COL in gdf.columns:
        mask = gdf[TARGET_COL].notna()
        y_true = pd.to_numeric(gdf.loc[mask, TARGET_COL], errors='coerce')
        y_pred = gdf.loc[mask, 'predicted']
        metrics['mae'] = float(mean_absolute_error(y_true.fillna(0), y_pred.fillna(0)))
        try:
            metrics['rmse'] = float(mean_squared_error(y_true.fillna(0), y_pred.fillna(0), squared=False))
        except TypeError:
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true.fillna(0), y_pred.fillna(0))))
    else:
        metrics['mae'] = None
        metrics['rmse'] = None

    # Save outputs
    pred_geo_path = OUT_DIR / 'predictions.geojson'
    try:
        gdf.to_file(pred_geo_path, driver='GeoJSON')
        print(f"Saved predictions to {pred_geo_path}")
    except Exception:
        print("Failed to write GeoJSON; skipping geometry export.")

    metrics_path = OUT_DIR / 'eval_metrics.json'
    metrics['n_rows'] = int(len(gdf))
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {metrics_path}")

    # Simple maps (choropleth saved as PNG)
    try:
        # predicted map
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        gdf.plot(column='predicted', ax=ax, legend=True, cmap='viridis', missing_kwds={"color": "lightgrey"})
        ax.set_title('Predicted livability score')
        plt.axis('off')
        p_pred = OUT_DIR / 'predicted_map.png'
        fig.savefig(p_pred, bbox_inches='tight', dpi=150)
        plt.close(fig)

        # residual map (if available)
        if 'residual' in gdf.columns and gdf['residual'].notna().any():
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            # cap residuals for visualization
            r = gdf['residual'].clip(-50, 50)
            gdf.assign(_r=r).plot(column='_r', ax=ax, legend=True, cmap='bwr', missing_kwds={"color": "lightgrey"})
            ax.set_title('Prediction residual (pred - actual)')
            plt.axis('off')
            p_res = OUT_DIR / 'residual_map.png'
            fig.savefig(p_res, bbox_inches='tight', dpi=150)
            plt.close(fig)
    except Exception as e:
        print(f"Failed to create maps: {e}")

    print("Evaluation complete")


if __name__ == '__main__':
    evaluate()
