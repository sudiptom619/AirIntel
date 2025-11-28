import geopandas as gpd
from rasterstats import zonal_stats
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Callable

try:
    import folium
except Exception:
    folium = None  # folium is optional and used only for interactive debugging


def compute_ndvi_features(ndvi_raster_path: str = "data/interim/ndvi_clipped.tif",
                          grid_path: str = "data/processed/city_grid.geojson") -> gpd.GeoDataFrame:
    """Compute NDVI zonal statistics for each grid cell and return a GeoDataFrame with NDVI features.

    Steps performed:
    - load grid and reproject to EPSG:3857 (matches many web-mercator rasters)
    - compute zonal stats (mean, median, min, max, std)
    - join stats back to grid
    - compute normalized NDVI (0-1) from ndvi_mean and clip to [0,1]
    - persist intermediate outputs (GeoJSON + parquet of non-geometry features)

    Note: This function assumes the raster and vector are in compatible CRS. If your raster is in a
    different CRS, either reproject the raster or read the grid in the raster CRS before running.
    """

    # Load grid and ensure projection matches common raster projections (web mercator)
    grid = gpd.read_file(grid_path)
    grid = grid.to_crs("EPSG:3857")

    # Read raster nodata explicitly to avoid rasterstats NodataWarning
    nod = None
    try:
        with rasterio.open(ndvi_raster_path) as src:
            nod = src.nodata
    except Exception:
        # If raster can't be opened here, let zonal_stats attempt to infer nodata
        nod = None

    # Compute zonal statistics
    zs = zonal_stats(
        vectors=grid.geometry,
        raster=ndvi_raster_path,
        stats=["mean", "median", "min", "max", "std"],
        all_touched=True,
        nodata=nod,
    )

    # Convert to DataFrame and rename columns to explicit feature names
    ndvi_df = pd.DataFrame(zs).rename(columns={
        "mean": "ndvi_mean",
        "median": "ndvi_median",
        "min": "ndvi_min",
        "max": "ndvi_max",
        "std": "ndvi_std",
    })

    # Join zonal stats back to the GeoDataFrame
    grid = grid.reset_index(drop=True).join(ndvi_df)

    # Normalization: NDVI is in [-1, 1] -> map to [0, 1] for ML and scoring
    # Negative NDVI typically indicates water/barren/urban; this mapping preserves ordering.
    grid["ndvi_norm"] = (grid["ndvi_mean"] + 1) / 2
    grid["ndvi_norm"] = grid["ndvi_norm"].clip(0, 1)

    # Persist outputs
    grid.to_file("data/processed/city_grid_with_ndvi.geojson", driver="GeoJSON")
    grid.drop(columns="geometry").to_parquet("data/processed/features_ndvi.parquet")

    return grid


def integrate_features(grid: Optional[gpd.GeoDataFrame] = None,
                       add_population_fn: Optional[Callable] = None,
                       add_elevation_fn: Optional[Callable] = None,
                       add_osm_fn: Optional[Callable] = None,
                       add_aqi_fn: Optional[Callable] = None) -> gpd.GeoDataFrame:
    """Integrate NDVI with other feature augmentation functions and save final feature store.

    The add_* functions are expected to take and return a GeoDataFrame. They are optional and
    integration will call whichever are provided.
    """

    if grid is None:
        grid = compute_ndvi_features()

    # Call optional augmentation functions if provided
    if add_population_fn is not None:
        grid = add_population_fn(grid)
    if add_elevation_fn is not None:
        grid = add_elevation_fn(grid)
    if add_osm_fn is not None:
        grid = add_osm_fn(grid)
    if add_aqi_fn is not None:
        grid = add_aqi_fn(grid)

    # Save final feature store (geometry kept in GeoParquet or drop geometry for pure table)
    # Use GeoDataFrame.to_parquet in recent geopandas or drop geometry before saving with pyarrow
    try:
        grid.to_parquet("data/processed/features.parquet")
    except Exception:
        # Fallback: drop geometry and save with pandas
        grid.drop(columns="geometry").to_parquet("data/processed/features.parquet")

    return grid


def plot_ndvi_histogram(grid: gpd.GeoDataFrame, bins: int = 30):
    """Quick QC: histogram of ndvi_mean values."""
    if "ndvi_mean" not in grid.columns:
        raise ValueError("ndvi_mean column not found in grid; run compute_ndvi_features first")
    ax = grid["ndvi_mean"].hist(bins=bins)
    ax.set_title("NDVI mean histogram")
    ax.set_xlabel("ndvi_mean")
    ax.set_ylabel("count")
    plt.tight_layout()
    return ax


def folium_ndvi_map(grid: gpd.GeoDataFrame, center: Optional[tuple] = None, zoom_start: int = 12):
    """Create a folium choropleth for ndvi_mean. Returns a folium.Map object.

    If folium is not installed this will raise ImportError.
    """
    if folium is None:
        raise ImportError("folium is not installed; install it to use folium_ndvi_map")

    # Ensure grid in WGS84 for folium and expose the index as a property
    grid_wgs = grid.to_crs("EPSG:4326").reset_index()

    if center is None:
        centroid = grid_wgs.unary_union.centroid
        center = (centroid.y, centroid.x)

    m = folium.Map(location=list(center), zoom_start=zoom_start)

    # Use the reset index (named 'index') as the join key in feature properties
    # key_on='feature.properties.index' will match the 'index' property created by reset_index()
    folium.Choropleth(
        geo_data=grid_wgs.__geo_interface__,
        name="NDVI",
        data=grid_wgs,
        columns=["index", "ndvi_mean"],
        key_on="feature.properties.index",
        fill_color="YlGn",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="NDVI mean",
    ).add_to(m)

    return m


if __name__ == "__main__":
    # Minimal runnable sanity check when module executed directly
    g = compute_ndvi_features()
    try:
        plot_ndvi_histogram(g)
    except Exception:
        pass
