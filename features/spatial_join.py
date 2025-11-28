# features/spatial_join.py
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import argparse

def compute_osm_features(grid_fp="data/processed/city_grid.geojson",
                         roads_fp="data/interim/roads.geojson",
                         inds_fp="data/interim/industrial.geojson",
                         out_grid_fp="data/processed/city_grid_with_osm.geojson",
                         out_parquet="data/processed/features.parquet"):
    # load
    grid = gpd.read_file(grid_fp)  # in EPSG:4326
    roads = gpd.read_file(roads_fp)  # EPSG:4326
    inds = gpd.read_file(inds_fp)    # EPSG:4326
    # project everything to metric CRS for length/distance (EPSG:3857)
    metric = "EPSG:3857"
    grid_m = grid.to_crs(metric)
    roads_m = roads.to_crs(metric)
    inds_m = inds.to_crs(metric)
    # ensure roads are lines (drop invalid)
    roads_m = roads_m[roads_m.geometry.type.isin(["LineString","MultiLineString"])].copy()
    # compute road length inside each grid cell
    grid_m['road_length_m'] = 0.0
    for idx, cell in grid_m.iterrows():
        try:
            clipped = roads_m.intersection(cell.geometry)
            # sum lengths
            total = 0.0
            for geom in clipped:
                if geom is None or geom.is_empty:
                    continue
                total += geom.length
            grid_m.at[idx, 'road_length_m'] = total
        except Exception as e:
            grid_m.at[idx, 'road_length_m'] = 0.0
    # compute density
    grid_m['road_density_m_per_m2'] = grid_m['road_length_m'] / grid_m['area_m2']
    # compute centroid distances to nearest industrial polygon
    # prepare centroids
    centroids = grid_m.centroid
    # union industrial polygons for faster distance calc
    if not inds_m.empty:
        inds_union = inds_m.unary_union
        dists = [centroid.distance(inds_union) if not inds_union.is_empty else float('nan') for centroid in centroids]
    else:
        dists = [float('nan')]*len(centroids)
    grid_m['dist_to_industry_m'] = dists
    # reproject back to WGS84 and save
    grid_out = grid_m.to_crs("EPSG:4326")
    grid_out.to_file(out_grid_fp, driver="GeoJSON")
    # build features parquet
    df = grid_out[['cell_id','centroid_lat','centroid_lon','road_length_m','road_density_m_per_m2','dist_to_industry_m','area_m2','geometry']].copy()
    df.to_parquet(out_parquet, index=False)
    print("Saved grid with OSM features to", out_grid_fp)
    print("Saved features to", out_parquet)

if __name__ == "__main__":
    compute_osm_features()
