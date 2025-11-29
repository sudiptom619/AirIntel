"""Train a livability model with spatial (block) cross-validation.

Produces a saved model at data/models/livability_model.pkl and a CV predictions
file at data/models/cv_predictions.parquet. Uses the feature store produced by
`features/scoring.py` (`data/processed/features.parquet`).

Usage: python models/train.py
Optional flags (edit constants below or add argparser as needed):
 - TARGET_COL: target column name in feature store (default: 'livability_score')
 - BLOCK_SIZE_M: spatial block size in meters used to form groups for GroupKFold
"""
import os
import math
import json
from pathlib import Path

import joblib
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import xgboost as xgb
except Exception:
    xgb = None

ROOT = Path(__file__).resolve().parent.parent
FEATURE_STORE = ROOT / "data" / "processed" / "features.parquet"
MODEL_DIR = ROOT / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "livability_model.pkl"
CV_PRED_PATH = MODEL_DIR / "cv_predictions.parquet"

# Configurable defaults
TARGET_COL = "livability_score"
# Candidate feature sets: prefer raw environmental features; if unavailable,
# fall back to component scores produced by features/scoring.py
RAW_FEATURES = [
    "pm25",
    "pm10",
    "road_density",
    "dist_to_industry",
    "ndvi_mean",
    "pop_density",
    "elev_mean",
    "flood_proxy",
]

COMPONENT_FEATURES = [
    "pollution_score",
    "traffic_score",
    "industry_score",
    "green_score",
    "pop_score",
    "flood_score",
    "elev_score",
]

# Initial feature_cols is None so prepare_X_y can auto-detect which set is present
FEATURE_COLS = None
N_SPLITS = 5
BLOCK_SIZE_M = 10000  # 10 km grid blocks for spatial grouping; tune if needed


def load_features(path=FEATURE_STORE):
    if not Path(path).exists():
        raise FileNotFoundError(f"Feature store not found: {path}")
    try:
        gdf = gpd.read_parquet(path)
    except Exception:
        gdf = gpd.read_file(path)
    # ensure metric centroids exist
    if 'centroid_x_m' in gdf.columns and 'centroid_y_m' in gdf.columns:
        gdf['centroid_x_m'] = pd.to_numeric(gdf['centroid_x_m'], errors='coerce')
        gdf['centroid_y_m'] = pd.to_numeric(gdf['centroid_y_m'], errors='coerce')
    else:
        try:
            gdf_m = gdf.to_crs(epsg=3857)
            gdf['centroid_x_m'] = gdf_m.geometry.centroid.x
            gdf['centroid_y_m'] = gdf_m.geometry.centroid.y
        except Exception:
            # last resort: centroid in degree space (less accurate)
            gdf['centroid_x_m'] = gdf.geometry.centroid.x
            gdf['centroid_y_m'] = gdf.geometry.centroid.y
    return gdf


def try_build_features_from_parts():
    """Attempt to assemble a richer feature store from partial processed artifacts.

    This looks for known intermediate files (city_grid_with_osm.geojson,
    data/processed/features_ndvi.parquet, data/processed/pollution_by_grid.parquet)
    and merges them. If successful, writes FEATURE_STORE and returns True.
    """
    from pathlib import Path
    from importlib import import_module

    parts = []
    root = Path(__file__).resolve().parent.parent
    try:
        # prefer grid with OSM if available
        grid_path = root / 'data' / 'processed' / 'city_grid_with_osm.geojson'
        if not grid_path.exists():
            grid_path = root / 'data' / 'processed' / 'city_grid.geojson'
        g = gpd.read_file(grid_path)
    except Exception:
        return False

    # try attach NDVI
    ndvi_path = root / 'data' / 'processed' / 'features_ndvi.parquet'
    if ndvi_path.exists():
        try:
            import pandas as pd
            nd = pd.read_parquet(ndvi_path)
            # merge on cell_id
            g = g.merge(nd[["cell_id", "ndvi_mean"]], on='cell_id', how='left')
        except Exception:
            pass

    # try attach pollution
    pol_path = root / 'data' / 'processed' / 'pollution_by_grid.parquet'
    if pol_path.exists():
        try:
            import pandas as pd
            pol = pd.read_parquet(pol_path)
            # ensure geometry not duplicated
            g = g.merge(pol[['cell_id', 'pm25', 'pm10']], on='cell_id', how='left')
        except Exception:
            pass

    # normalize column names expected by scoring
    if 'dist_to_industry_m' in g.columns and 'dist_to_industry' not in g.columns:
        g['dist_to_industry'] = g['dist_to_industry_m']
    if 'road_length_m' in g.columns and 'road_density' not in g.columns:
        # scoring expects road_length_m and area_m2; keep road_length_m
        g['road_length_m'] = g['road_length_m']

    # Try to compute components and livability using features.scoring
    try:
        # import features.scoring by file path to avoid package import issues when
        # running this script directly
        scoring_path = root / 'features' / 'scoring.py'
        import importlib.util
        spec = importlib.util.spec_from_file_location('features.scoring', scoring_path)
        scoring = importlib.util.module_from_spec(spec)
        # ensure repo root on sys.path so relative/package imports inside scoring work
        import sys
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        spec.loader.exec_module(scoring)
        g2 = scoring.compute_components(g)
        weights = scoring.load_feature_config()
        g3 = scoring.compute_livability(g2, weights=weights)
        # save feature store
        try:
            g3.to_parquet(FEATURE_STORE, index=False)
        except Exception:
            g3.drop(columns='geometry', errors='ignore').to_parquet(FEATURE_STORE, index=False)
        print(f"Re-built feature store from parts and wrote: {FEATURE_STORE}")
        return True
    except Exception as e:
        print(f"Failed to auto-build features from parts: {e}")
        return False


def make_block_ids(gdf, block_size_m=BLOCK_SIZE_M):
    # floor-divide centroids to make blocks
    x = gdf['centroid_x_m'].fillna(0).astype(float)
    y = gdf['centroid_y_m'].fillna(0).astype(float)
    bx = (np.floor_divide(x, block_size_m)).astype(int)
    by = (np.floor_divide(y, block_size_m)).astype(int)
    blocks = (bx.astype(str) + "_" + by.astype(str))
    return blocks


def prepare_X_y(gdf, feature_cols=FEATURE_COLS, target_col=TARGET_COL):
    # Determine target
    if target_col not in gdf.columns:
        raise KeyError(f"Target column '{target_col}' not found in feature store")
    df = gdf.copy()

    # Auto-detect available features
    chosen = None
    if feature_cols and isinstance(feature_cols, (list, tuple)):
        chosen = [f for f in feature_cols if f in df.columns]

    if not chosen:
        # prefer raw features if available
        raw_present = [f for f in RAW_FEATURES if f in df.columns]
        if len(raw_present) >= max(3, int(len(RAW_FEATURES) / 2)):
            chosen = raw_present
        else:
            comp_present = [f for f in COMPONENT_FEATURES if f in df.columns]
            if len(comp_present) > 0:
                chosen = comp_present

    if not chosen or len(chosen) == 0:
        raise KeyError(
            "No usable features found in feature store. Expected one of raw features or component score columns."
        )

    print(f"Using features for training: {chosen}")

    X = df[chosen].copy()
    # Fill missing numeric features with 0 (consistent with predict module)
    X = X.fillna(0).astype(float)
    y = pd.to_numeric(df[target_col], errors='coerce')
    return X, y, df


def train_and_cv(X, y, groups, n_splits=N_SPLITS):
    folds = GroupKFold(n_splits=n_splits)
    preds = np.full(len(y), np.nan)
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(folds.split(X, y, groups=groups)):
        print(f"Training fold {fold + 1}/{n_splits} — train={len(train_idx)} val={len(val_idx)}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if xgb is not None:
            model = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=4,
                verbosity=0
            )
            # early stopping using validation set
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=25,
                verbose=False
            )
        else:
            # fallback to scikit-learn RandomForest if xgboost missing
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=200, n_jobs=4, random_state=42)
            model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        preds[val_idx] = y_pred
        mae = mean_absolute_error(y_val.fillna(0), np.nan_to_num(y_pred, nan=0.0))
        # compute RMSE (some sklearn versions may not accept squared kwarg)
        try:
            rmse = mean_squared_error(y_val.fillna(0), np.nan_to_num(y_pred, nan=0.0), squared=False)
        except TypeError:
            rmse = float(np.sqrt(mean_squared_error(y_val.fillna(0), np.nan_to_num(y_pred, nan=0.0))))
        fold_metrics.append({'fold': fold, 'mae': float(mae), 'rmse': float(rmse)})
        print(f" Fold {fold}: MAE={mae:.3f} RMSE={rmse:.3f}")

    return preds, fold_metrics


def fit_final_model(X, y):
    if xgb is not None:
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=4,
            verbosity=0
        )
        model.fit(X, y.fillna(0))
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=300, n_jobs=4, random_state=42)
        model.fit(X, y.fillna(0))
    return model


def main():
    print("Loading feature store:", FEATURE_STORE)
    gdf = load_features(FEATURE_STORE)
    print(f"Rows in feature store: {len(gdf)}")
    # If component scores exist but are all NaN, attempt to rebuild from processed parts
    comp_cols = ['pollution_score','traffic_score','industry_score','green_score','pop_score','flood_score','elev_score']
    if all(c in gdf.columns for c in comp_cols):
        non_null_counts = sum(gdf[c].notna().sum() for c in comp_cols)
        if non_null_counts == 0:
            print('Component score columns present but all NaN — attempting to rebuild feature store from intermediate artifacts')
            ok = try_build_features_from_parts()
            if ok:
                # reload
                gdf = load_features(FEATURE_STORE)
            else:
                print('Auto-build from parts failed; proceeding with existing feature store')
    groups = make_block_ids(gdf)
    X, y, df = prepare_X_y(gdf)

    # If the target column is entirely missing but component scores exist,
    # compute a livability target deterministically (same logic as features.scoring)
    if y.notna().sum() == 0:
        try:
            from features import scoring
            print("Target column appears empty — computing livability_score from component scores using features.scoring.compute_livability()")
            df2 = scoring.compute_livability(df)
            # update df and y
            df = df2
            y = pd.to_numeric(df[TARGET_COL], errors='coerce')
            # recompute X from chosen features (prepare_X_y expects gdf)
            X = df[[c for c in X.columns if c in df.columns]].fillna(0).astype(float)
        except Exception as e:
            print(f"Failed to compute livability target from components: {e}")
            # proceed and allow later check to raise
    # drop rows with missing target
    valid_mask = y.notna()
    Xv = X.loc[valid_mask].reset_index(drop=True)
    yv = y.loc[valid_mask].reset_index(drop=True)
    groups_v = groups.loc[valid_mask].reset_index(drop=True)
    df_valid = df.loc[valid_mask].reset_index(drop=True)

    print(f"Training on {len(Xv)} rows after dropping missing targets")

    # Basic sanity checks before CV: need at least n_splits samples and at least
    # n_splits distinct groups for GroupKFold
    n_samples = len(Xv)
    n_groups = groups_v.nunique() if hasattr(groups_v, 'nunique') else len(set(groups_v))
    if n_samples < N_SPLITS or n_groups < N_SPLITS:
        msg = (
            f"Not enough labeled data to run GroupKFold with n_splits={N_SPLITS}.\n"
            f"Found n_samples={n_samples}, n_groups={n_groups}.\n"
            "Possible causes:\n"
            " - The feature store does not contain a target column ('livability_score') or it's all NaN.\n"
            " - Component score columns are missing or NaN, so an automatic livability target couldn't be computed.\n\n"
            "Next steps you can take:\n"
            " 1) Run the feature pipeline to compute component scores and livability:\n"
            "     python -m features.scoring\n"
            "     or: python features/scoring.py\n"
            " 2) Inspect `data/processed/features.parquet` to ensure component columns like 'pollution_score' exist and have values.\n"
            " 3) If you have a separate target (e.g., PM2.5), add or point TARGET_COL to that column.\n"
            " 4) Reduce N_SPLITS in this script to a smaller number (e.g., 2) if you have very few groups.\n"
        )
        raise RuntimeError(msg)

    preds, fold_metrics = train_and_cv(Xv, yv, groups_v, n_splits=N_SPLITS)

    # attach CV predictions
    df_valid['predicted_cv'] = preds
    df_valid['predicted_cv_residual'] = df_valid['predicted_cv'] - df_valid[TARGET_COL]
    # Save CV predictions
    try:
        df_valid.to_parquet(CV_PRED_PATH, index=False)
        print(f"Saved CV predictions to {CV_PRED_PATH}")
    except Exception:
        print("Failed to save CV predictions to parquet; skipping.")

    # Print aggregate metrics
    mae_all = mean_absolute_error(yv.fillna(0), np.nan_to_num(preds, nan=0.0))
    try:
        rmse_all = mean_squared_error(yv.fillna(0), np.nan_to_num(preds, nan=0.0), squared=False)
    except TypeError:
        rmse_all = float(np.sqrt(mean_squared_error(yv.fillna(0), np.nan_to_num(preds, nan=0.0))))
    print("CV fold metrics:")
    for fm in fold_metrics:
        print(json.dumps(fm))
    print(f"Aggregate CV MAE={mae_all:.3f} RMSE={rmse_all:.3f}")

    # Fit a final model on all available rows (including those without target? prefer only with target)
    final_model = fit_final_model(Xv, yv)
    joblib.dump(final_model, MODEL_PATH)
    print(f"Saved final model to {MODEL_PATH}")


if __name__ == '__main__':
    main()
