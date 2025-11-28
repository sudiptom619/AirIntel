# etl/fetch_osm.py
import requests, json
import geopandas as gpd
from shapely.geometry import shape
import argparse
import time

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def query_overpass(bbox):
    # bbox: south,west,north,east
    s,w,n,e = bbox
    # Query: highways (roads) and industrial landuse polygons
    q = f"""
    [out:json][timeout:60];
    (
      way["highway"]({s},{w},{n},{e});
      way["landuse"="industrial"]({s},{w},{n},{e});
      relation["landuse"="industrial"]({s},{w},{n},{e});
    );
    out body;
    >;
    out skel qt;
    """
    r = requests.post(OVERPASS_URL, data={'data': q})
    r.raise_for_status()
    return r.json()

def osm_to_gdfs(osm_json):
    # Convert Overpass-style JSON to GeoDataFrames for ways (lines/polygons)
    elements = osm_json.get("elements", [])
    nodes = {el['id']: el for el in elements if el['type'] == 'node'}
    ways = [el for el in elements if el['type'] == 'way']
    lines = []
    polys = []
    for way in ways:
        coords = []
        for nid in way['nodes']:
            node = nodes.get(nid)
            if node:
                coords.append((node['lon'], node['lat']))
        if 'landuse' in way.get('tags', {}) and way['tags'].get('landuse') == 'industrial':
            # polygon
            try:
                # ensure closed ring
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                poly = {"type":"Feature", "geometry":{"type":"Polygon", "coordinates":[coords]}, "properties": way.get('tags', {})}
                polys.append(poly)
            except Exception as e:
                continue
        else:
            # road/line
            line = {"type":"Feature", "geometry":{"type":"LineString", "coordinates": coords}, "properties": way.get('tags', {})}
            lines.append(line)
    roads_gdf = gpd.GeoDataFrame.from_features(lines, crs="EPSG:4326")
    inds_gdf = gpd.GeoDataFrame.from_features(polys, crs="EPSG:4326")
    return roads_gdf, inds_gdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", type=str, required=True)  # "s,w,n,e"
    parser.add_argument("--out_raw", default="data/raw/osm/osm_raw.json")
    parser.add_argument("--out_clean", default="data/interim/cleaned_osm.geojson")
    args = parser.parse_args()
    s,w,n,e = [float(x) for x in args.bbox.split(",")]
    print("Querying Overpass for bbox", (s,w,n,e))
    osm = query_overpass((s,w,n,e))
    with open(args.out_raw, "w") as f:
        json.dump(osm, f)
    print("Saved raw OSM JSON to", args.out_raw)
    roads, inds = osm_to_gdfs(osm)
    # save separate
    if not roads.empty:
        roads.to_file("data/interim/roads.geojson", driver="GeoJSON")
    if not inds.empty:
        inds.to_file("data/interim/industrial.geojson", driver="GeoJSON")
    print("Saved roads and industrial to data/interim/")
