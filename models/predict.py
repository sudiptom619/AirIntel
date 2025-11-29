# models/predict.py
import joblib
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, mapping
import os
import math

FEATURE_STORE = "data/processed/features.parquet"  # output of integrate_features()
MODEL_PATH = "data/models/livability_model.pkl"    # saved model (optional; scoring.py fallback)

_model = None
_features_gdf = None

def load_model():
    """Lazy-load model if available; return None otherwise."""
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
        else:
            _model = None
    return _model

def load_features():
    """
    Load the feature GeoDataFrame once and ensure:
     - it's a GeoDataFrame
     - it has a geometry column (EPSG:4326)
     - a precomputed centroid in metric CRS (EPSG:3857) named '_centroid_m'
    """
    global _features_gdf
    if _features_gdf is None:
        try:
            _features_gdf = gpd.read_parquet(FEATURE_STORE)
        except Exception:
            _features_gdf = gpd.read_file(FEATURE_STORE)

        if not isinstance(_features_gdf, gpd.GeoDataFrame):
            _features_gdf = gpd.GeoDataFrame(_features_gdf, geometry="geometry", crs="EPSG:4326")

        # Ensure geometry CRS exists
        if _features_gdf.crs is None:
            _features_gdf.set_crs(epsg=4326, inplace=True)

        # Precompute metric centroids for faster distance calculations
        if "_centroid_m" not in _features_gdf.columns:
            try:
                gdf_m = _features_gdf.to_crs(epsg=3857)
                _features_gdf["_centroid_m"] = gdf_m.geometry.centroid
            except Exception:
                # fallback: compute centroid in 4326 (deg) — still works but distances will be inaccurate
                _features_gdf["_centroid_m"] = _features_gdf.geometry.centroid

    return _features_gdf

def find_nearest_cell(lat, lon):
    """
    Return (index_label, row_series, distance_in_meters)
    Finds the nearest grid cell centroid to the provided lat/lon.
    """
    gdf = load_features()
    # Create point in metric CRS
    try:
        pt_m = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
    except Exception:
        # If reprojection fails, use lat/lon point (less accurate)
        pt_m = Point(lon, lat)

    # Compute distances against precomputed metric centroids
    centroids = gdf["_centroid_m"]
    # centroids is a GeoSeries — its .distance can take a shapely geometry
    dists = centroids.distance(pt_m)
    idx = dists.idxmin()
    dist_m = float(dists.loc[idx]) if pd.notna(dists.loc[idx]) else float("nan")

    row = gdf.loc[idx].copy()
    return idx, row, dist_m

def impute_features(row: pd.Series, features_list):
    """
    Non-destructive imputation: returns a copy of row with missing feature columns filled
    with column medians computed from the feature store.
    """
    gdf = load_features()
    available = [f for f in features_list if f in gdf.columns]
    medians = pd.Series(dtype=float)
    if available:
        medians = gdf[available].median()

    row_copy = row.copy()
    for f in features_list:
        if f not in row_copy.index or pd.isna(row_copy.get(f)):
            if f in medians.index and pd.notna(medians[f]):
                row_copy[f] = medians[f]
            else:
                # leave as NaN for now; later we'll fill remaining NaNs with 0 in feature vector
                row_copy[f] = np.nan
    return row_copy

def build_feature_vector_for_point(lat, lon, feature_cols=None):
    """
    Returns: (idx_label, X_array (1xN), row_copy (Series), dist_m)
    """
    idx, row, dist_m = find_nearest_cell(lat, lon)

    if feature_cols is None:
        feature_cols = ["pm25","pm10","road_density","dist_to_industry",
                        "ndvi_mean","pop_density","elev_mean","flood_proxy"]

    row_copy = impute_features(row, feature_cols)

    # Ensure all expected feature columns exist
    for f in feature_cols:
        if f not in row_copy.index:
            row_copy[f] = np.nan

    # Prepare X: replace remaining NaNs with 0 (models generally expect numeric)
    X = row_copy[feature_cols].fillna(0).astype(float).values.reshape(1, -1)
    return idx, X, row_copy, dist_m

def predict_point(lat, lon, return_components=True):
    """
    Predict livability score for a point. Returns a JSON-friendly dict.
    """
    model = load_model()
    idx, X, row, dist_m = build_feature_vector_for_point(lat, lon)

    # Predict or fallback to deterministic scoring
    score = None
    raw_pred = None
    if model is not None:
        try:
            raw_pred = model.predict(X)[0]
            if raw_pred is None or (isinstance(raw_pred, float) and (math.isnan(raw_pred) or math.isinf(raw_pred))):
                score = None
            else:
                # Robust scaling:
                if 0.0 <= raw_pred <= 1.0:
                    score = float(raw_pred * 100.0)
                elif 1.0 < raw_pred <= 100.0:
                    score = float(raw_pred)
                else:
                    # If model outputs unexpected scale, clip into 0..100
                    score = float(np.clip(raw_pred, 0.0, 100.0))
        except Exception:
            score = None

    if score is None:
        # fallback deterministic scoring
        score = deterministic_score_from_row(row)

    # Build components: prefer precomputed *_score columns, otherwise compute
    comps = {}
    precomputed_keys = {
        'pollution_score': 'pollution_score',
        'traffic_score': 'traffic_score',
        'industry_score': 'industry_score',
        'green_score': 'green_score',
        'pop_score': 'pop_score',
        'flood_score': 'flood_score',
        'elev_score': 'elev_score'
    }

    has_any_precomputed = any(k in row.index and pd.notna(row.get(k)) for k in precomputed_keys.values())
    if has_any_precomputed:
        for out_key, colname in precomputed_keys.items():
            comps[out_key] = safe(row.get(colname))
    else:
        raw_comps = compute_component_scores(row)
        comps = {
            'pollution_score': safe(raw_comps.get('pollution')),
            'traffic_score': safe(raw_comps.get('traffic')),
            'industry_score': safe(raw_comps.get('industry')),
            'green_score': safe(raw_comps.get('green')),
            'pop_score': safe(raw_comps.get('population')),
            'flood_score': safe(raw_comps.get('flood')),
            'elev_score': safe(row.get('elev_mean'))  # optional: convert elevation to a score if desired
        }

    # Compute confidence: prefer 'confidence' column if present; otherwise fraction of non-missing component scores
    confidence = None
    if 'confidence' in row.index and pd.notna(row.get('confidence')):
        try:
            confidence = float(row.get('confidence'))
        except Exception:
            confidence = None
    if confidence is None:
        vals = [v for v in comps.values()]
        non_missing = sum(1 for v in vals if v is not None)
        confidence = (non_missing / len(vals)) if len(vals) > 0 else 0.0

    # cell_id and centroid handling (sanitized)
    cell_id = row.get('cell_id') if 'cell_id' in row.index else None

    centroid = None
    if 'geometry' in row.index and row.geometry is not None:
        try:
            c = row.geometry.centroid
            centroid = (float(c.y), float(c.x))
        except Exception:
            centroid = None

    # Build JSON-friendly feature snapshot: convert geometry -> WKT or geojson mapping
    feature_snapshot = row.to_dict()
    # replace geometry with WKT or mapping
    if 'geometry' in feature_snapshot:
        try:
            geom = feature_snapshot['geometry']
            if geom is not None:
                feature_snapshot['geometry'] = mapping(geom)
            else:
                feature_snapshot['geometry'] = None
        except Exception:
            feature_snapshot['geometry'] = None

    # Convert numpy types to native python types where possible
    def pyify(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    feature_snapshot = {k: pyify(v) for k, v in feature_snapshot.items()}

    # Prepare index output: return original idx label and an int version only if convertible
    cell_index_int = None
    try:
        cell_index_int = int(idx)
    except Exception:
        cell_index_int = None

    result = {
        "lat": float(lat),
        "lon": float(lon),
        "cell_index": idx,
        "cell_index_int": cell_index_int,
        "distance_to_cell_m": float(dist_m) if (dist_m is not None and not math.isnan(dist_m)) else None,
        "score": round(float(np.clip(score, 0.0, 100.0)), 2),
        "components": comps,
        "confidence": round(float(confidence), 3) if confidence is not None else None,
        "cell_id": cell_id,
        "centroid": centroid,
        "feature_snapshot": feature_snapshot
    }
    return result

# Deterministic scoring and component builders (kept similar to your originals, but robustified)
def deterministic_score_from_row(row):
    w = {"pollution": 0.4, "traffic": 0.2, "green": 0.15, "industry": 0.1, "flood": 0.1, "population": 0.05}
    comps = compute_component_scores(row)
    s = 0.0
    for k, wt in w.items():
        val = comps.get(k)
        if val is None or pd.isna(val):
            # reasonable default if component missing
            val = 50.0
        s += float(val) * float(wt)
    return float(np.clip(s, 0.0, 100.0))

def compute_component_scores(row):
    # pollution_score: lower pm25 -> higher score
    pm = row.get("pm25", np.nan)
    if pd.isna(pm):
        pollution_score = 60.0
    else:
        try:
            pollution_score = np.interp(pm, [0, 35, 75, 150], [100, 80, 40, 0])
        except Exception:
            pollution_score = 60.0

    # traffic_score: lower road_density -> higher score
    rd = row.get("road_density", np.nan)
    try:
        traffic_score = np.interp(rd if not pd.isna(rd) else 0.0, [0, 0.001, 0.01], [100, 70, 30])
    except Exception:
        traffic_score = 50.0

    # green_score: ndvi -1..1 -> 0..100
    ndvi = row.get("ndvi_mean", np.nan)
    if pd.isna(ndvi):
        green_score = 50.0
    else:
        try:
            if -1.0 <= ndvi <= 1.0:
                green_score = ((ndvi + 1.0) / 2.0) * 100.0
            else:
                green_score = float(np.clip(ndvi * 100.0, 0.0, 100.0))
        except Exception:
            green_score = 50.0

    # industry_score: farther is better
    dist_ind = row.get("dist_to_industry", np.nan)
    try:
        industry_score = np.interp(dist_ind if not pd.isna(dist_ind) else 20000.0,
                                   [0, 500, 2000, 10000], [0, 30, 70, 100])
    except Exception:
        industry_score = 50.0

    # flood_score: inverse flood_proxy (higher proxy -> worse)
    flood = row.get("flood_proxy", np.nan)
    try:
        flood_score = np.interp(flood if not pd.isna(flood) else 0.0, [0, 1, 5, 20], [100, 80, 50, 0])
    except Exception:
        flood_score = 50.0

    # population: lower pop density -> better
    pop = row.get("pop_density", np.nan)
    try:
        pop_score = np.interp(pop if not pd.isna(pop) else 1000.0, [0, 1000, 5000, 20000], [100, 70, 40, 20])
    except Exception:
        pop_score = 50.0

    comps = {
        "pollution": round(float(np.clip(pollution_score, 0.0, 100.0)), 2),
        "traffic": round(float(np.clip(traffic_score, 0.0, 100.0)), 2),
        "green": round(float(np.clip(green_score, 0.0, 100.0)), 2),
        "industry": round(float(np.clip(industry_score, 0.0, 100.0)), 2),
        "flood": round(float(np.clip(flood_score, 0.0, 100.0)), 2),
        "population": round(float(np.clip(pop_score, 0.0, 100.0)), 2)
    }
    return comps

def safe(x):
    """Return python float or None for numpy/pandas values"""
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        if pd.isna(x):
            return None
        if isinstance(x, np.generic):
            return float(x.item())
        if isinstance(x, (int, float)):
            return float(x)
        return float(x)
    except Exception:
        return None


if __name__ == '__main__':
    # Small CLI: python models/predict.py --lat 22.57 --lon 88.36
    import argparse, json

    parser = argparse.ArgumentParser(description='Predict livability score for a point')
    parser.add_argument('--lat', type=float, required=True, help='Latitude')
    parser.add_argument('--lon', type=float, required=True, help='Longitude')
    parser.add_argument('--no-components', action='store_true', help='Do not include component breakdown')
    args = parser.parse_args()
    res = predict_point(args.lat, args.lon, return_components=not args.no_components)
    # ensure JSON-serializable (predict_point already pyifies feature_snapshot and numeric types)
    print(json.dumps(res, indent=2))
