"""Feature normalization, scoring, and feature store utilities for Phase 5.

This module reads a GeoDataFrame of grid cells with raw feature columns and
produces: component scores (0-100), a livability_score (0-100), a confidence
value and writes a parquet feature store plus a lightweight JSONL cache for UI.

Drop this file into `features/` and run `python -m features.scoring` or
`python features/scoring.py` from the repository root to produce
`data/processed/features.parquet` and `data/processed/comparison_cache/comparison_cache.jsonl`.
"""
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import warnings

try:
    import yaml
except Exception:
    yaml = None
    warnings.warn("PyYAML not installed; will use default weights from code. To customize weights, install pyyaml.")

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
    warnings.warn("matplotlib not installed; histogram plotting will be skipped.")
from models import predict

ROOT = Path(__file__).resolve().parent.parent
FEATURE_PATH = ROOT / "data" / "processed" / "features.parquet"
CACHE_DIR = ROOT / "data" / "processed" / "comparison_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# optionally warm the model/feature cache in models.predict if available
try:
    if hasattr(predict, 'load_features'):
        predict.load_features()
    elif hasattr(predict, '_load_features'):
        predict._load_features()
except Exception:
    # Do not fail import if warming the cache fails; pipeline will load on demand
    pass


def clamp(x, lo=0.0, hi=100.0):
    try:
        return float(max(lo, min(hi, x)))
    except Exception:
        return np.nan


def safe_float(x):
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def pollution_score(pm25):
    """Piecewise mapping from PM2.5 (µg/m3) to 0-100 where higher is better.

    Uses commonly-cited breakpoints but clipped to [0,200].
    """
    if pd.isna(pm25):
        return np.nan
    x = float(pm25)
    x = max(0.0, min(x, 200.0))
    # piecewise to emphasize low concentrations
    if x <= 12:
        score = 100.0
    elif x <= 35:
        score = 85.0 + (35.0 - x) / (35.0 - 12.0) * 15.0
    elif x <= 55:
        score = 60.0 + (55.0 - x) / (55.0 - 35.0) * 25.0
    elif x <= 150:
        score = 20.0 + (150.0 - x) / (150.0 - 55.0) * 40.0
    else:
        score = 0.0
    return round(clamp(score, 0, 100), 2)


def traffic_score(road_length_m, area_m2):
    """Transform road length (meters inside cell) and area (m2) -> 0-100.

    Returns higher=better (less road density -> higher score).
    """
    try:
        if pd.isna(road_length_m) or pd.isna(area_m2) or area_m2 <= 0:
            return np.nan
    except Exception:
        return np.nan

    road_per_km2 = (float(road_length_m) / float(area_m2)) * 1e6
    # domain: 0..20000 m/km2 -> 100..0
    val = 100.0 * (1.0 - min(road_per_km2 / 20000.0, 1.0))
    return round(clamp(val, 0, 100), 2)


def industry_score(dist_to_ind_m):
    """Distance (m) to nearest industrial polygon -> 0-100 (farther = better).
    Cap distance at 5000 m.
    """
    if pd.isna(dist_to_ind_m):
        return np.nan
    d = float(dist_to_ind_m)
    dcap = min(max(d, 0.0), 5000.0)
    return round(clamp(100.0 * (dcap / 5000.0), 0, 100), 2)


def green_score(ndvi_mean):
    if pd.isna(ndvi_mean):
        return np.nan
    nd = float(ndvi_mean)
    nd_clamped = max(0.0, min(nd, 0.8))
    return round(clamp(100.0 * (nd_clamped / 0.8), 0, 100), 2)


def pop_score(pop_density, min_log=None, max_log=None):
    if pd.isna(pop_density):
        return np.nan
    plog = np.log1p(float(pop_density))
    # if min_log/max_log provided, use them; otherwise use reasonable defaults
    if min_log is None:
        min_log = 0.0
    if max_log is None:
        max_log = 10.0
    norm = (plog - min_log) / max(1e-9, (max_log - min_log))
    val = 100.0 * (1.0 - max(0.0, min(1.0, norm)))
    return round(clamp(val, 0, 100), 2)


def flood_score(flood_proxy, flood_proxy_95pct=1.0):
    if pd.isna(flood_proxy):
        return np.nan
    fp = float(flood_proxy)
    denom = flood_proxy_95pct if flood_proxy_95pct > 0 else 1.0
    frac = max(0.0, min(1.0, fp / denom))
    return round(clamp(100.0 * (1.0 - frac), 0, 100), 2)


def elev_score(elev, elev_min=None, elev_max=None):
    if pd.isna(elev):
        return np.nan
    e = float(elev)
    if elev_min is None or elev_max is None or elev_max <= elev_min:
        # fallback: simple clamp to 0..100
        return round(clamp(min(100.0, max(0.0, e))), 2)
    eclamped = min(max(e, elev_min), elev_max)
    val = 100.0 * ((eclamped - elev_min) / max(1e-9, (elev_max - elev_min)))
    return round(clamp(val, 0, 100), 2)


DEFAULT_WEIGHTS = {
    'pollution_score': 0.30,
    'traffic_score'  : 0.15,
    'industry_score' : 0.15,
    'green_score'    : 0.15,
    'pop_score'      : 0.10,
    'flood_score'    : 0.10,
    'elev_score'     : 0.05
}


CONFIG_PATH = ROOT / 'features' / 'feature_config.yaml'


def load_feature_config(path: Path = None):
    """Load feature weights from YAML config file.

    Returns a dict mapping component score column names to weights.
    If YAML isn't available or file missing, returns DEFAULT_WEIGHTS.
    """
    path = Path(path) if path is not None else CONFIG_PATH
    if not path.exists() or yaml is None:
        return DEFAULT_WEIGHTS.copy()
    try:
        cfg = yaml.safe_load(path.read_text())
        w = cfg.get('weights', {}) if isinstance(cfg, dict) else {}
        # map config keys to component score keys
        mapping = {
            'pollution': 'pollution_score',
            'traffic': 'traffic_score',
            'industry': 'industry_score',
            'green': 'green_score',
            'population': 'pop_score',
            'flood': 'flood_score',
            'elev': 'elev_score'
        }
        out = DEFAULT_WEIGHTS.copy()
        for k, val in w.items():
            mapped = mapping.get(k)
            if mapped:
                try:
                    out[mapped] = float(val)
                except Exception:
                    pass
        # normalize to sum 1.0 if weights provide only main components
        total = sum(out.values())
        if total > 0:
            for kk in out:
                out[kk] = out[kk] / total
        return out
    except Exception:
        return DEFAULT_WEIGHTS.copy()


def compute_components(gdf, pop_log_min=None, pop_log_max=None, flood_95=None):
    """Given a GeoDataFrame with raw columns, compute component scores.

    Expected raw column names (best-effort):
      - pm25
      - road_length_m
      - area_m2 (if missing, computed from geometry)
      - dist_to_industry_m or dist_to_industry
      - ndvi_mean
      - pop_density
      - elev_mean
      - flood_proxy

    Returns a copy of gdf with added *_score columns.
    """
    df = gdf.copy()

    # ensure area_m2 exists (compute in metric CRS)
    if 'area_m2' not in df.columns or df['area_m2'].isnull().any():
        try:
            df = df.set_geometry('geometry')
            df['area_m2'] = df.geometry.to_crs(epsg=3857).area
        except Exception:
            # If geometry missing or cannot project, try using provided area or set NaN
            df['area_m2'] = df.get('area_m2', pd.Series([np.nan]*len(df)))

    # canonicalize distance column names
    if 'dist_to_industry_m' in df.columns and 'dist_to_industry' not in df.columns:
        df['dist_to_industry'] = df['dist_to_industry_m']

    # compute population log bounds if not provided
    if pop_log_min is None or pop_log_max is None:
        if 'pop_density' in df.columns:
            # dropna then ensure numeric before log1p
            plogs = np.log1p(df['pop_density'].dropna().astype(float))
            if len(plogs) > 0:
                pop_log_min = pop_log_min if pop_log_min is not None else float(plogs.min())
                pop_log_max = pop_log_max if pop_log_max is not None else float(plogs.quantile(0.95))
            else:
                pop_log_min, pop_log_max = 0.0, 10.0
        else:
            pop_log_min, pop_log_max = 0.0, 10.0

    # flood 95th percentile
    if flood_95 is None and 'flood_proxy' in df.columns:
        flood_95 = float(df['flood_proxy'].dropna().quantile(0.95)) if df['flood_proxy'].dropna().size > 0 else 1.0

    df['pollution_score'] = df['pm25'].apply(pollution_score) if 'pm25' in df.columns else np.nan
    df['traffic_score'] = df.apply(lambda r: traffic_score(r.get('road_length_m', np.nan), r.get('area_m2', np.nan)), axis=1)
    df['industry_score'] = df['dist_to_industry'].apply(industry_score) if 'dist_to_industry' in df.columns else np.nan
    df['green_score'] = df['ndvi_mean'].apply(green_score) if 'ndvi_mean' in df.columns else np.nan
    df['pop_score'] = df['pop_density'].apply(lambda x: pop_score(x, pop_log_min, pop_log_max)) if 'pop_density' in df.columns else np.nan
    df['flood_score'] = df['flood_proxy'].apply(lambda x: flood_score(x, flood_95)) if 'flood_proxy' in df.columns else np.nan

    elev_min = float(df['elev_mean'].min()) if 'elev_mean' in df.columns and df['elev_mean'].dropna().size>0 else None
    elev_max = float(df['elev_mean'].max()) if 'elev_mean' in df.columns and df['elev_mean'].dropna().size>0 else None
    if elev_min is None or elev_max is None or elev_max <= elev_min:
        # fallback defaults
        elev_min, elev_max = 0.0, elev_min + 100.0 if elev_min is not None else 100.0
    df['elev_score'] = df['elev_mean'].apply(lambda e: elev_score(e, elev_min, elev_max)) if 'elev_mean' in df.columns else np.nan

    return df


def compute_livability(df, weights=DEFAULT_WEIGHTS, apply_penalties=True):
    df = df.copy()
    comps = list(weights.keys())

    # ensure component columns exist
    for c in comps:
        if c not in df.columns:
            df[c] = np.nan

    def row_score(row):
        weighted_vals = []
        used_weights = []
        for comp, wt in weights.items():
            v = row.get(comp, np.nan)
            if not pd.isna(v):
                weighted_vals.append(float(v) * float(wt))
                used_weights.append(float(wt))
        if len(used_weights) == 0:
            return np.nan, 0.0
        raw = sum(weighted_vals) / sum(used_weights)
        # optional penalty: if pollution_score < 20 reduce by 8%
        if apply_penalties and row.get('pollution_score', 100) is not None and not pd.isna(row.get('pollution_score')):
            if float(row.get('pollution_score')) < 20.0:
                raw = raw * 0.92
        return round(clamp(raw, 0, 100), 2), round(sum(used_weights), 3)

    scores = df.apply(lambda r: pd.Series(row_score(r), index=['livability_score', 'score_weight_sum']), axis=1)
    df['livability_score'] = scores['livability_score']
    df['score_weight_sum'] = scores['score_weight_sum']
    df['confidence'] = df[[c for c in comps]].notna().mean(axis=1)
    df['last_updated'] = datetime.utcnow().isoformat()
    return df


def save_feature_store(df, path: Path = FEATURE_PATH):
    df_out = df.copy()
    # prefer to preserve geometry; GeoPandas will write geometry into parquet if supported
    path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(path, index=False)

    # export a compact JSONL cache for UI quick loads
    cache_file = CACHE_DIR / "comparison_cache.jsonl"
    with open(cache_file, 'w', encoding='utf8') as f:
        for _, row in df_out.iterrows():
            geom = row.geometry if 'geometry' in row.index else None
            centroid = None
            if geom is not None and hasattr(geom, 'centroid'):
                centroid = geom.centroid
            obj = {
                'cell_id': row.get('cell_id'),
                'lon': float(centroid.x) if centroid is not None else safe_float(row.get('centroid_lon')),
                'lat': float(centroid.y) if centroid is not None else safe_float(row.get('centroid_lat')),
                'livability_score': safe_float(row.get('livability_score')),
                'confidence': safe_float(row.get('confidence')),
                'components': {
                    'pollution_score': safe_float(row.get('pollution_score')),
                    'traffic_score': safe_float(row.get('traffic_score')),
                    'industry_score': safe_float(row.get('industry_score')),
                    'green_score': safe_float(row.get('green_score')),
                    'pop_score': safe_float(row.get('pop_score')),
                    'flood_score': safe_float(row.get('flood_score')),
                    'elev_score': safe_float(row.get('elev_score'))
                }
            }
            f.write(json.dumps(obj) + "\n")
    print(f"Feature store written to: {path}\nCache written to: {cache_file}")


def run_pipeline(grid_path: Path = None, out_path: Path = None):
    """High-level runner: read a city grid (geojson or parquet), compute components,
    livability and write feature store and cache."""
    grid_path = Path(grid_path) if grid_path is not None else ROOT / "data" / "processed" / "city_grid.geojson"
    out_path = Path(out_path) if out_path is not None else FEATURE_PATH
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")
    print(f"Reading grid from {grid_path}")
    gdf = gpd.read_file(grid_path)
    # ensure geometry
    if gdf.geometry.is_empty.all():
        raise ValueError("Input grid has no geometry")
    # compute components
    gdf2 = compute_components(gdf)
    # load weights from config if available
    weights = load_feature_config()
    gdf3 = compute_livability(gdf2, weights=weights)

    # ensure centroid lon/lat columns (EPSG:4326)
    try:
        gdf3['centroid_lon'] = gdf3.geometry.centroid.x
        gdf3['centroid_lat'] = gdf3.geometry.centroid.y
    except Exception:
        gdf3['centroid_lon'] = gdf3.get('centroid_lon', pd.NA)
        gdf3['centroid_lat'] = gdf3.get('centroid_lat', pd.NA)

    # compute metric centroids (EPSG:3857) for faster distance queries
    try:
        gdf_m = gdf3.to_crs(epsg=3857)
        cent_x = gdf_m.geometry.centroid.x
        cent_y = gdf_m.geometry.centroid.y
        gdf3['centroid_x_m'] = cent_x
        gdf3['centroid_y_m'] = cent_y
    except Exception:
        gdf3['centroid_x_m'] = pd.NA
        gdf3['centroid_y_m'] = pd.NA

    # Save feature store and cache
    save_feature_store(gdf3, out_path)

    # Save a small sample rows CSV for inspection
    sample_path = Path(out_path).parent / 'features_sample.csv'
    try:
        gdf3.drop(columns=[c for c in ['geometry'] if c in gdf3.columns]).head(200).to_csv(sample_path, index=False)
    except Exception:
        gdf3.head(200).to_csv(sample_path, index=False)

    # Plot histograms for livability and components (if matplotlib available)
    if plt is not None:
        try:
            plot_dir = Path(out_path).parent
            plt.figure(figsize=(6,4))
            gdf3['livability_score'].dropna().hist(bins=30)
            plt.title('Livability score distribution')
            plt.xlabel('livability_score')
            plt.savefig(plot_dir / 'livability_hist.png')
            plt.close()
            # per-component
            comps = ['pollution_score','traffic_score','industry_score','green_score','pop_score','flood_score','elev_score']
            for comp in comps:
                if comp in gdf3.columns:
                    plt.figure(figsize=(6,3))
                    gdf3[comp].dropna().hist(bins=30)
                    plt.title(comp)
                    plt.savefig(plot_dir / f'{comp}_hist.png')
                    plt.close()
        except Exception:
            pass


if __name__ == '__main__':
    # run the pipeline using defaults
    run_pipeline()
