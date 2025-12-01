# tests/test_etl.py
"""Tests for ETL modules."""
import pytest
import tempfile
import json
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box


def test_osm_to_gdfs_basic():
    """Test that osm_to_gdfs returns non-empty GeoDataFrame for valid input."""
    from etl.fetch_osm import osm_to_gdfs
    
    # Minimal mock OSM JSON with a few nodes and a way
    mock_osm = {
        "elements": [
            {"type": "node", "id": 1, "lat": 22.57, "lon": 88.36},
            {"type": "node", "id": 2, "lat": 22.58, "lon": 88.37},
            {"type": "node", "id": 3, "lat": 22.57, "lon": 88.38},
            {
                "type": "way", 
                "id": 100, 
                "nodes": [1, 2, 3],
                "tags": {"highway": "primary"}
            }
        ]
    }
    
    roads, inds = osm_to_gdfs(mock_osm)
    
    # Should have at least one road
    assert isinstance(roads, gpd.GeoDataFrame)
    assert len(roads) >= 1
    
    # Industrial should be empty for this input
    assert isinstance(inds, gpd.GeoDataFrame)


def test_osm_to_gdfs_with_industrial():
    """Test that industrial polygons are correctly parsed."""
    from etl.fetch_osm import osm_to_gdfs
    
    # Mock OSM with industrial polygon
    mock_osm = {
        "elements": [
            {"type": "node", "id": 1, "lat": 22.57, "lon": 88.36},
            {"type": "node", "id": 2, "lat": 22.58, "lon": 88.36},
            {"type": "node", "id": 3, "lat": 22.58, "lon": 88.37},
            {"type": "node", "id": 4, "lat": 22.57, "lon": 88.37},
            {
                "type": "way",
                "id": 200,
                "nodes": [1, 2, 3, 4, 1],  # closed ring
                "tags": {"landuse": "industrial"}
            }
        ]
    }
    
    roads, inds = osm_to_gdfs(mock_osm)
    
    # Industrial should have one polygon
    assert isinstance(inds, gpd.GeoDataFrame)
    assert len(inds) == 1


def test_osm_parsing_overpass_to_gdf():
    """Test the OSM parsing utility."""
    from etl.utils.osm_parsing import overpass_to_gdf
    
    mock_osm = {
        "elements": [
            {"type": "node", "id": 1, "lat": 22.57, "lon": 88.36},
            {"type": "node", "id": 2, "lat": 22.58, "lon": 88.37},
            {"type": "node", "id": 3, "lat": 22.59, "lon": 88.38},
            {
                "type": "way",
                "id": 100,
                "nodes": [1, 2, 3],
                "tags": {"highway": "residential"}
            }
        ]
    }
    
    gdf = overpass_to_gdf(mock_osm)
    
    assert isinstance(gdf, gpd.GeoDataFrame)
    # CRS may be None if empty or set if non-empty
    if not gdf.empty:
        assert gdf.crs is not None


def test_api_client_basic():
    """Test basic API client initialization."""
    from etl.utils.api_clients import SimpleRequestClient
    
    client = SimpleRequestClient(retries=2, backoff=0.5, timeout=10)
    
    assert client.retries == 2
    assert client.backoff == 0.5
    assert client.timeout == 10


def test_population_estimation():
    """Test population density estimation."""
    from etl.fetch_population import estimate_population_density
    
    # Create a simple grid
    polys = [box(88.35 + i * 0.01, 22.55 + j * 0.01, 
                 88.36 + i * 0.01, 22.56 + j * 0.01) 
             for i in range(3) for j in range(3)]
    
    grid = gpd.GeoDataFrame({
        'cell_id': [f'cell_{i}' for i in range(9)],
        'geometry': polys
    }, crs="EPSG:4326")
    
    result = estimate_population_density(grid)
    
    assert 'pop_density' in result.columns
    assert result['pop_density'].notna().all()
    assert (result['pop_density'] > 0).all()


def test_elevation_estimation():
    """Test elevation estimation."""
    from etl.fetch_elevation import estimate_elevation
    
    # Create a simple grid
    polys = [box(88.35 + i * 0.01, 22.55 + j * 0.01,
                 88.36 + i * 0.01, 22.56 + j * 0.01)
             for i in range(3) for j in range(3)]
    
    grid = gpd.GeoDataFrame({
        'cell_id': [f'cell_{i}' for i in range(9)],
        'geometry': polys
    }, crs="EPSG:4326")
    
    result = estimate_elevation(grid)
    
    assert 'elev_mean' in result.columns
    assert result['elev_mean'].notna().all()


def test_flood_proxy_computation():
    """Test flood proxy computation from elevation."""
    from etl.fetch_elevation import compute_flood_proxy
    
    # Create grid with elevation
    polys = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)]
    grid = gpd.GeoDataFrame({
        'cell_id': ['a', 'b', 'c'],
        'elev_mean': [5.0, 15.0, 30.0],  # Different elevations
        'geometry': polys
    }, crs="EPSG:4326")
    
    result = compute_flood_proxy(grid)
    
    assert 'flood_proxy' in result.columns
    # Lower elevation should have higher flood risk
    assert result.loc[0, 'flood_proxy'] > result.loc[2, 'flood_proxy']
