import sys
from pathlib import Path

# Ensure repository root is on sys.path so we can import local packages
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from features.engineer_features import compute_ndvi_features, folium_ndvi_map


if __name__ == "__main__":
    g = compute_ndvi_features()
    m = folium_ndvi_map(g)
    out = "data/processed/ndvi_map.html"
    m.save(out)
    print(out)
