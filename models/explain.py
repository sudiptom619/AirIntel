# models/explain.py
import joblib
import shap
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "data" / "models" / "livability_model.pkl"
FEATURE_STORE = ROOT / "data" / "processed" / "features.parquet"
OUT_DIR = ROOT / "data" / "models" / "shap"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_shap_plots(feature_cols=None, model_path=MODEL_PATH, feature_store=FEATURE_STORE, out_dir=OUT_DIR):
    """Generate SHAP summary and a dependence plot for the top feature.

    Saves PNGs into out_dir.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not Path(feature_store).exists():
        raise FileNotFoundError(f"Feature store not found: {feature_store}")

    model = joblib.load(model_path)
    gdf = gpd.read_parquet(feature_store)

    if feature_cols is None:
        preferred = ["pm25", "pm10", "road_density", "dist_to_industry", "ndvi_mean", "pop_density", "elev_mean", "flood_proxy"]
        # try model feature names first
        try:
            if hasattr(model, 'feature_names_in_'):
                feature_cols = list(model.feature_names_in_)
        except Exception:
            feature_cols = None
        if not feature_cols:
            feature_cols = [c for c in preferred if c in gdf.columns]

    if len(feature_cols) == 0:
        raise RuntimeError('No usable feature columns found for SHAP explanation')

    X = gdf[feature_cols].fillna(0)

    # Try modern SHAP API; fallback to TreeExplainer
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        # shap_values may be an explanation object; pass to summary_plot
        plt.figure(figsize=(8, 6))
        shap.plots.bar(shap_values, max_display=20, show=False)
        p_summary = out_dir / "shap_summary.png"
        plt.savefig(p_summary, bbox_inches="tight", dpi=150)
        plt.close()

        # dependence plot for top feature
        top_feat = X.columns[0]
        try:
            plt.figure(figsize=(6, 5))
            shap.plots.scatter(shap_values[:, top_feat], color=shap_values[:, top_feat].values, show=False)
            p_dep = out_dir / f"shap_dependence_{top_feat}.png"
            plt.savefig(p_dep, bbox_inches="tight", dpi=150)
            plt.close()
        except Exception:
            # best-effort only
            pass

    except Exception:
        # fallback for older shap versions
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        p_summary = out_dir / "shap_summary.png"
        plt.savefig(p_summary, bbox_inches="tight", dpi=150)
        plt.close()

        top_feat = X.columns[0]
        try:
            plt.figure(figsize=(6, 5))
            shap.dependence_plot(top_feat, shap_values, X, show=False)
            p_dep = out_dir / f"shap_dependence_{top_feat}.png"
            plt.savefig(p_dep, bbox_inches="tight", dpi=150)
            plt.close()
        except Exception:
            pass


if __name__ == '__main__':
    generate_shap_plots()
