# etl/fetch_elevation.py
"""
Fetch and process elevation data for the grid.

Uses SRTM or similar elevation data. For demo, generates synthetic elevation 
if no raster is available.
"""
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats

RAW_DIR = Path("data/raw/elevation")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = Path("data/processed")


def estimate_elevation(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Estimate elevation for each grid cell.
    
    Creates a synthetic elevation model based on position.
    In production, replace with actual SRTM data.
    
    Args:
        grid: GeoDataFrame with grid cells
        
    Returns:
        Grid with elev_mean and slope columns added
    """
    grid = grid.copy()
    
    try:
        # Get centroids
        centroids = grid.geometry.centroid
        lats = centroids.y.values
        lons = centroids.x.values
        
        # Create synthetic elevation: base + gradient + noise
        # Higher elevation in the north, lower in south (typical for many areas)
        np.random.seed(42)
        
        # Base elevation around 10m (coastal city like Kolkata)
        base_elev = 10
        
        # Latitude gradient: +5m per 0.1 degree north
        lat_gradient = (lats - lats.min()) * 50
        
        # Longitude gradient: slight variation
        lon_gradient = (lons - lons.min()) * 20
        
        # Random variation (±5m)
        noise = np.random.normal(0, 5, len(grid))
        
        grid['elev_mean'] = (base_elev + lat_gradient + lon_gradient + noise).clip(0, 200)
        
        # Compute approximate slope based on neighboring differences
        # For demo, use random slope values
        grid['slope'] = np.abs(np.random.normal(2, 2, len(grid))).clip(0, 15)
        
        print("Generated synthetic elevation data")
        
    except Exception as e:
        print(f"Warning: Could not estimate elevation: {e}")
        np.random.seed(42)
        grid['elev_mean'] = np.random.uniform(5, 30, len(grid))
        grid['slope'] = np.random.uniform(0, 10, len(grid))
    
    return grid


def add_elevation_from_raster(grid: gpd.GeoDataFrame,
                               raster_path: str = None) -> gpd.GeoDataFrame:
    """
    Add elevation data from a raster file using zonal statistics.
    
    Args:
        grid: GeoDataFrame with grid cells
        raster_path: Path to elevation raster (SRTM/DEM format)
        
    Returns:
        Grid with elev_mean and elev_std columns added
    """
    grid = grid.copy()
    
    if raster_path is None or not Path(raster_path).exists():
        print("No elevation raster found, using synthetic estimates")
        return estimate_elevation(grid)
    
    try:
        # Ensure grid is in the same CRS as raster (usually EPSG:4326 for SRTM)
        grid_wgs = grid.to_crs(epsg=4326)
        
        # Compute zonal stats
        stats = zonal_stats(
            grid_wgs.geometry,
            raster_path,
            stats=['mean', 'min', 'max', 'std'],
            all_touched=True
        )
        
        # Add to grid
        elev_df = pd.DataFrame(stats)
        grid['elev_mean'] = elev_df['mean'].fillna(0)
        grid['elev_min'] = elev_df['min'].fillna(0)
        grid['elev_max'] = elev_df['max'].fillna(0)
        grid['elev_std'] = elev_df['std'].fillna(0)
        
        # Estimate slope from elevation range
        grid['slope'] = ((grid['elev_max'] - grid['elev_min']) / 1000).clip(0, 45)
        
        print(f"Elevation computed from raster: {raster_path}")
        
    except Exception as e:
        print(f"Could not load elevation raster: {e}. Using synthetic estimates.")
        return estimate_elevation(grid)
    
    return grid


def compute_flood_proxy(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute a simple flood risk proxy based on elevation.
    
    Lower elevation = higher flood risk.
    
    Args:
        grid: GeoDataFrame with elev_mean column
        
    Returns:
        Grid with flood_proxy column added
    """
    grid = grid.copy()
    
    if 'elev_mean' not in grid.columns:
        grid['flood_proxy'] = 0.5
        return grid
    
    try:
        elev = grid['elev_mean'].fillna(grid['elev_mean'].median())
        
        # Normalize elevation: lower = higher flood risk
        elev_min = elev.min()
        elev_range = elev.max() - elev_min
        if elev_range <= 0:
            elev_range = 1
        
        # Flood proxy: 0 (no risk) to 1 (high risk)
        # Lower elevation = higher risk
        grid['flood_proxy'] = 1 - (elev - elev_min) / elev_range
        
        # Boost flood risk for very low elevations (below 5m)
        grid.loc[grid['elev_mean'] < 5, 'flood_proxy'] = grid.loc[
            grid['elev_mean'] < 5, 'flood_proxy'
        ].clip(lower=0.7)
        
    except Exception as e:
        print(f"Could not compute flood proxy: {e}")
        grid['flood_proxy'] = 0.5
    
    return grid


def main():
    """Process elevation data for the grid."""
    parser = argparse.ArgumentParser(description="Fetch and process elevation data")
    parser.add_argument("--grid", default="data/processed/city_grid.geojson",
                       help="Path to grid GeoJSON file")
    parser.add_argument("--raster", default=None,
                       help="Path to elevation raster (optional)")
    parser.add_argument("--out", default="data/processed/city_grid_with_elev.geojson",
                       help="Output path")
    args = parser.parse_args()
    
    print(f"Loading grid from {args.grid}")
    grid = gpd.read_file(args.grid)
    
    grid = add_elevation_from_raster(grid, args.raster)
    grid = compute_flood_proxy(grid)
    
    grid.to_file(args.out, driver="GeoJSON")
    print(f"Saved grid with elevation to {args.out}")
    
    # Also save as parquet for faster loading
    parquet_path = args.out.replace('.geojson', '.parquet')
    try:
        grid.to_parquet(parquet_path)
        print(f"Saved to {parquet_path}")
    except Exception:
        grid.drop(columns='geometry').to_parquet(parquet_path)
        print(f"Saved (without geometry) to {parquet_path}")


if __name__ == "__main__":
    main()
