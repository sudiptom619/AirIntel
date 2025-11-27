# app/utils/geo.py
import geopandas as gpd
from shapely.geometry import box
import pyproj

def bbox_from_center(lat, lon, delta_deg=0.05):
    """
    Return bbox in (south, west, north, east) given center and delta degrees.
    Useful quick helper; prefer bbox in meters in production.
    """
    return (lat - delta_deg, lon - delta_deg, lat + delta_deg, lon + delta_deg)

def create_square_grid(bbox, resolution_m=1000, crs_epsg=3857):
    """
    Create a square grid of polygons over bbox (lat/lon in EPSG:4326).
    Steps:
     - create a projected bbox in meters (EPSG:3857) for uniform cell size
     - create square polygons of size resolution_m
     - return GeoDataFrame in EPSG:4326
    """
    # Convert bbox to projected CRS
    bbox_polygon = box(bbox[1], bbox[0], bbox[3], bbox[2])  # lon/lat order
    gdf = gpd.GeoDataFrame({'geometry': [bbox_polygon]}, crs="EPSG:4326")
    gdf_proj = gdf.to_crs(epsg=crs_epsg)
    minx, miny, maxx, maxy = gdf_proj.total_bounds

    xs = list(range(int(minx), int(maxx) + resolution_m, resolution_m))
    ys = list(range(int(miny), int(maxy) + resolution_m, resolution_m))

    cells = []
    for x in xs:
        for y in ys:
            cells.append(box(x, y, x + resolution_m, y + resolution_m))

    grid = gpd.GeoDataFrame({'geometry': cells}, crs=f"EPSG:{crs_epsg}")
    grid = grid.to_crs(epsg=4326)  # back to lat/lon for storage/visualization
    grid["cell_id"] = grid.index.astype(str)
    return grid
