# features/make_grid.py
import geopandas as gpd
from shapely.geometry import box
import pyproj
from functools import partial
from shapely.ops import transform
import json
import argparse

def create_grid(bbox, resolution_m=1000, crs_epsg=3857):
    #(south, west, north, east)
    s, w, n, e = bbox

    wgs84 = "EPSG:4326"
    metric = f"EPSG:{crs_epsg}"

    bbox_poly = box(w, s, e, n)
    gdf = gpd.GeoDataFrame({'geometry':[bbox_poly]}, crs=wgs84)
    gdf = gdf.to_crs(metric)
    bounds = gdf.total_bounds  
    minx, miny, maxx, maxy = bounds


    # build squares
    x_coords = list(range(int(minx), int(maxx), int(resolution_m)))
    y_coords = list(range(int(miny), int(maxy), int(resolution_m)))
    polys = []
    ids = []
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            poly = box(x, y, x + resolution_m, y + resolution_m)
            polys.append(poly)
            ids.append(f"{i}_{j}")
    grid = gpd.GeoDataFrame({'cell_id': ids, 'geometry': polys}, crs=metric)


    # clip
    grid = gpd.overlay(grid, gdf.to_crs(metric), how='intersection')


    # compute centroid 
    grid['area_m2'] = grid.geometry.area
    grid_wgs = grid.to_crs(wgs84)
    grid_wgs['centroid_lon'] = grid_wgs.centroid.x
    grid_wgs['centroid_lat'] = grid_wgs.centroid.y
    return grid_wgs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", type=str, required=True,
                        help="south,west,north,east (comma separated)")
    parser.add_argument("--res", type=int, default=1000)
    parser.add_argument("--out", type=str, default="data/processed/city_grid.geojson")
    args = parser.parse_args()
    s,w,n,e = [float(x) for x in args.bbox.split(",")]
    grid = create_grid((s,w,n,e), resolution_m=args.res)
    grid.to_file(args.out, driver="GeoJSON")
    print("Saved grid to", args.out)
