# etl/fetch_population.py
"""
Fetch and process population density data for the grid.

Uses WorldPop or similar publicly available population raster data.
For demo purposes, we generate synthetic population data based on road density
and distance to city center, since WorldPop requires registration for bulk downloads.
"""
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats

RAW_DIR = Path("data/raw/population")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = Path("data/processed")


def estimate_population_density(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Estimate population density for each grid cell.
    
    Uses a simple model based on distance to city center and road density 
    (if available). In production, replace with actual WorldPop raster.
    
    Args:
        grid: GeoDataFrame with grid cells
        
    Returns:
        Grid with pop_density column added
    """
    grid = grid.copy()
    
    # Find approximate city center (centroid of all cells)
    try:
        all_centroids = grid.geometry.centroid
        center = all_centroids.unary_union.centroid
        
        # Project to metric CRS for distance calculation
        grid_m = grid.to_crs(epsg=3857)
        center_m = gpd.GeoSeries([center], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]
        
        # Calculate distance from each cell to center
        distances = grid_m.geometry.centroid.distance(center_m)
        
        # Normalize distances (0 = center, 1 = farthest)
        max_dist = distances.max() if distances.max() > 0 else 1
        norm_dist = distances / max_dist
        
        # Population model: higher near center, decreases with distance
        # Base population density: 5000 people/km² at center, 500 at edge
        base_pop = 5000 - 4500 * norm_dist
        
        # Add some random variation (±20%)
        np.random.seed(42)
        variation = 1 + (np.random.random(len(grid)) - 0.5) * 0.4
        
        # If road density available, use it to boost population
        if 'road_density_m_per_m2' in grid.columns:
            road_factor = 1 + grid['road_density_m_per_m2'].fillna(0) * 100
            road_factor = road_factor.clip(1, 2)
        else:
            road_factor = 1
        
        grid['pop_density'] = (base_pop * variation * road_factor).clip(100, 15000)
        
    except Exception as e:
        print(f"Warning: Could not estimate population density: {e}")
        # Fallback: uniform density
        np.random.seed(42)
        grid['pop_density'] = np.random.uniform(500, 5000, len(grid))
    
    return grid


def add_population_from_raster(grid: gpd.GeoDataFrame, 
                                raster_path: str = None) -> gpd.GeoDataFrame:
    """
    Add population density from a raster file using zonal statistics.
    
    Args:
        grid: GeoDataFrame with grid cells
        raster_path: Path to population raster (WorldPop format)
        
    Returns:
        Grid with pop_density column added
    """
    grid = grid.copy()
    
    if raster_path is None or not Path(raster_path).exists():
        print("No population raster found, using synthetic estimates")
        return estimate_population_density(grid)
    
    try:
        # Ensure grid is in the same CRS as raster (usually EPSG:4326 for WorldPop)
        grid_wgs = grid.to_crs(epsg=4326)
        
        # Compute zonal stats
        stats = zonal_stats(
            grid_wgs.geometry,
            raster_path,
            stats=['mean', 'sum'],
            all_touched=True
        )
        
        # Add to grid
        pop_df = pd.DataFrame(stats)
        grid['pop_density'] = pop_df['mean'].fillna(0)
        grid['pop_total'] = pop_df['sum'].fillna(0)
        
        print(f"Population density computed from raster: {raster_path}")
        
    except Exception as e:
        print(f"Could not load population raster: {e}. Using synthetic estimates.")
        return estimate_population_density(grid)
    
    return grid


def main():
    """Process population data for the grid."""
    parser = argparse.ArgumentParser(description="Fetch and process population data")
    parser.add_argument("--grid", default="data/processed/city_grid.geojson",
                       help="Path to grid GeoJSON file")
    parser.add_argument("--raster", default=None,
                       help="Path to population raster (optional)")
    parser.add_argument("--out", default="data/processed/city_grid_with_pop.geojson",
                       help="Output path")
    args = parser.parse_args()
    
    print(f"Loading grid from {args.grid}")
    grid = gpd.read_file(args.grid)
    
    grid = add_population_from_raster(grid, args.raster)
    
    grid.to_file(args.out, driver="GeoJSON")
    print(f"Saved grid with population to {args.out}")
    
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
