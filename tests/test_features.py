# tests/test_features.py
"""Tests for feature engineering modules."""
import pytest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point


def test_grid_creation():
    """Test that grid creation yields expected cell count."""
    from features.make_grid import create_grid
    
    # Small bounding box: roughly 2km x 2km
    # bbox = (south, west, north, east)
    bbox = (22.55, 88.35, 22.57, 88.37)  # About 2km x 2km
    resolution_m = 500  # 500m cells
    
    grid = create_grid(bbox, resolution_m=resolution_m)
    
    assert isinstance(grid, gpd.GeoDataFrame)
    assert len(grid) > 0
    assert 'cell_id' in grid.columns
    assert 'geometry' in grid.columns
    
    # With 500m resolution over ~2km, expect roughly 4x4 = 16 cells (give or take)
    assert 4 <= len(grid) <= 25


def test_grid_has_centroids():
    """Test that grid cells have centroid coordinates."""
    from features.make_grid import create_grid
    
    bbox = (22.55, 88.35, 22.57, 88.37)
    grid = create_grid(bbox, resolution_m=1000)
    
    assert 'centroid_lat' in grid.columns
    assert 'centroid_lon' in grid.columns
    assert grid['centroid_lat'].notna().all()
    assert grid['centroid_lon'].notna().all()


def test_grid_has_area():
    """Test that grid cells have area computed."""
    from features.make_grid import create_grid
    
    bbox = (22.55, 88.35, 22.57, 88.37)
    grid = create_grid(bbox, resolution_m=1000)
    
    assert 'area_m2' in grid.columns
    # Most cells should be around 1,000,000 m² (1km x 1km)
    # Edge cells may be smaller due to clipping
    assert grid['area_m2'].max() > 500000  # At least some cells are large
    assert (grid['area_m2'] > 0).all()  # All cells have positive area


def test_spatial_join_roads():
    """Test that spatial join computes road density."""
    from features.spatial_join import compute_osm_features
    
    # Create test data in temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create grid
        polys = [
            box(88.35, 22.55, 88.36, 22.56),
            box(88.36, 22.55, 88.37, 22.56)
        ]
        grid = gpd.GeoDataFrame({
            'cell_id': ['a', 'b'],
            'area_m2': [1000000, 1000000],
            'centroid_lat': [22.555, 22.555],
            'centroid_lon': [88.355, 88.365],
            'geometry': polys
        }, crs="EPSG:4326")
        grid_path = tmpdir / "grid.geojson"
        grid.to_file(grid_path, driver="GeoJSON")
        
        # Create roads (simple lines)
        from shapely.geometry import LineString
        roads = gpd.GeoDataFrame({
            'highway': ['primary', 'secondary'],
            'geometry': [
                LineString([(88.35, 22.55), (88.36, 22.56)]),
                LineString([(88.355, 22.55), (88.355, 22.56)])
            ]
        }, crs="EPSG:4326")
        roads_path = tmpdir / "roads.geojson"
        roads.to_file(roads_path, driver="GeoJSON")
        
        # Create industrial (simple polygon)
        inds = gpd.GeoDataFrame({
            'landuse': ['industrial'],
            'geometry': [box(88.38, 22.55, 88.39, 22.56)]
        }, crs="EPSG:4326")
        inds_path = tmpdir / "industrial.geojson"
        inds.to_file(inds_path, driver="GeoJSON")
        
        # Output paths
        out_grid = tmpdir / "grid_osm.geojson"
        out_parquet = tmpdir / "features.parquet"
        
        # Run spatial join
        compute_osm_features(
            grid_fp=str(grid_path),
            roads_fp=str(roads_path),
            inds_fp=str(inds_path),
            out_grid_fp=str(out_grid),
            out_parquet=str(out_parquet)
        )
        
        # Check output
        result = gpd.read_file(out_grid)
        assert 'road_length_m' in result.columns
        assert 'road_density_m_per_m2' in result.columns
        assert 'dist_to_industry_m' in result.columns


def test_feature_config_loading():
    """Test that feature config loads correctly."""
    from features.scoring import load_feature_config
    
    weights = load_feature_config()
    
    assert isinstance(weights, dict)
    assert len(weights) > 0
    # Should sum to approximately 1
    total = sum(weights.values())
    assert 0.99 <= total <= 1.01


def test_component_score_functions():
    """Test individual component scoring functions."""
    from features.scoring import (
        pollution_score, 
        traffic_score, 
        industry_score,
        green_score
    )
    
    # Pollution: lower PM2.5 = higher score
    assert pollution_score(5) > pollution_score(50)
    assert pollution_score(0) == 100
    
    # Traffic: less road = higher score
    assert traffic_score(0, 1000000) == 100
    assert traffic_score(1000, 1000000) < 100
    
    # Industry: farther = higher score
    assert industry_score(5000) > industry_score(100)
    assert industry_score(0) == 0
    
    # Green: higher NDVI = higher score
    assert green_score(0.8) > green_score(0.2)


def test_engineer_features_ndvi():
    """Test NDVI feature computation."""
    # This test requires a raster file, so we just test the function signature
    from features.engineer_features import compute_ndvi_features
    
    # Function should be callable
    assert callable(compute_ndvi_features)


def test_compute_livability_range():
    """Test that livability scores are in valid range."""
    from features.scoring import compute_components, compute_livability
    
    # Create test grid with various feature values
    polys = [box(0, 0, 1, 1), box(1, 0, 2, 1)]
    grid = gpd.GeoDataFrame({
        'cell_id': ['a', 'b'],
        'pm25': [10.0, 80.0],
        'road_length_m': [100, 2000],
        'area_m2': [1000000, 1000000],
        'dist_to_industry': [5000, 100],
        'ndvi_mean': [0.6, 0.1],
        'pop_density': [500, 8000],
        'elev_mean': [15, 5],
        'flood_proxy': [0.1, 0.9],
        'geometry': polys
    }, crs="EPSG:4326")
    
    comps = compute_components(grid)
    result = compute_livability(comps)
    
    assert 'livability_score' in result.columns
    assert (result['livability_score'] >= 0).all()
    assert (result['livability_score'] <= 100).all()
    
    # Cell 'a' should have higher score (better metrics)
    assert result.loc[0, 'livability_score'] > result.loc[1, 'livability_score']
