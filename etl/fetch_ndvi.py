import os
import geopandas as gpd

# use package-relative import (works when running as a module from the project root)
from .utils.raster_tools import clip_and_reproject


def main():
    grid = gpd.read_file("data/processed/city_grid.geojson")
    bbox = grid.total_bounds  # EPSG:4326

    src = "data/raw/ndvi/ndvi_kolkata.tif"
    out = "data/interim/ndvi_clipped.tif"
    
    os.makedirs("data/interim", exist_ok=True)

    clip_and_reproject(
        src_path=src,
        dst_path=out,
        bbox=bbox,
        dst_crs="EPSG:3857"       # metric CRS for zonal stats
    )

    print("Saved:", out)

if __name__ == "__main__":
    main()
