# app/utils/geo.py
"""Geospatial utility functions for the app."""
import geopandas as gpd
from pathlib import Path


def save_gjson(gdf: gpd.GeoDataFrame, path: str):
    """Save GeoDataFrame to GeoJSON file.
    
    Args:
        gdf: GeoDataFrame to save
        path: Output file path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON")


def bbox_from_center(lat: float, lon: float, delta_deg: float = 0.05):
    """
    Return bbox in (south, west, north, east) given center and delta degrees.
    
    Args:
        lat: Center latitude
        lon: Center longitude
        delta_deg: Half-width in degrees
        
    Returns:
        Tuple of (south, west, north, east)
    """
    return (lat - delta_deg, lon - delta_deg, lat + delta_deg, lon + delta_deg)


def meters_to_degrees(meters: float, lat: float = 0) -> float:
    """
    Approximate conversion from meters to degrees.
    
    This is a rough approximation that varies with latitude.
    
    Args:
        meters: Distance in meters
        lat: Latitude for correction (default 0 = equator)
        
    Returns:
        Approximate degrees
    """
    import math
    # At equator, 1 degree ≈ 111km
    # Adjust for latitude
    lat_rad = math.radians(lat)
    meters_per_deg = 111320 * math.cos(lat_rad)
    return meters / meters_per_deg if meters_per_deg > 0 else meters / 111320
