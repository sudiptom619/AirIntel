import math
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon

from features import scoring


def test_pollution_score_edges():
    assert scoring.pollution_score(0) == 100.0
    assert scoring.pollution_score(12) == 100.0
    # larger value should be lower score
    s1 = scoring.pollution_score(20)
    s2 = scoring.pollution_score(60)
    assert s1 > s2


def test_industry_score():
    assert scoring.industry_score(0) == 0.0
    assert scoring.industry_score(2500) == 50.0
    assert scoring.industry_score(6000) == 100.0


def test_traffic_score_and_area():
    # 0 roads -> 100
    assert scoring.traffic_score(0.0, 1000000.0) == 100.0
    # increasing road length reduces score
    s_small = scoring.traffic_score(100.0, 100000.0)  # some roads in small area
    s_large = scoring.traffic_score(1000.0, 100000.0)
    assert s_small > s_large


def test_compute_livability_confidence():
    # create a tiny GeoDataFrame with two cells, one with missing features
    geom = [Polygon([(0,0),(1,0),(1,1),(0,1)]), Polygon([(1,0),(2,0),(2,1),(1,1)])]
    df = gpd.GeoDataFrame({
        'cell_id': ['a','b'],
        'pm25': [10.0, None],
        'road_length_m': [100.0, None],
        'ndvi_mean': [0.2, None],
        'pop_density': [1000.0, None],
        'elev_mean': [10.0, None],
        'flood_proxy': [0.0, None]
    }, geometry=geom, crs='EPSG:4326')

    comps = scoring.compute_components(df)
    out = scoring.compute_livability(comps)
    assert 'livability_score' in out.columns
    assert out.loc[0, 'confidence'] > out.loc[1, 'confidence']
