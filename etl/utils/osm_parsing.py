# etl/utils/osm_parsing.py
import json
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon

def overpass_to_gdf(osm_json):
    """
    Convert a simple Overpass API JSON dump into GeoDataFrames.
    Note: Overpass output parsing can be complex; this helper supports common responses
    by assembling nodes/ways into geometries. For complex projects consider osmium or osmnx.
    """
    elements = osm_json.get('elements', [])
    nodes = {el['id']: (el['lon'], el['lat']) for el in elements if el['type']=='node'}
    ways = [el for el in elements if el['type'] == 'way']

    features = []
    for way in ways:
        coords = []
        for nid in way.get('nodes', []):
            if nid in nodes:
                coords.append(nodes[nid])
        if len(coords) < 2:
            continue

        tags = way.get('tags', {})
        geom = None
        # closed ring -> polygon
        if coords[0] == coords[-1] and len(coords) >= 4:
            geom = Polygon(coords)
            geom_type = "polygon"
        else:
            geom = LineString(coords)
            geom_type = "linestring"

        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": tags | {"osm_id": way.get('id'), "geom_type": geom_type}
        })

    # Build GeoDataFrame
    if not features:
        return gpd.GeoDataFrame(columns=['geometry', 'osm_id'], geometry='geometry', crs="EPSG:4326")

    # Build records list with geometry as a separate column
    records = []
    geometries = []
    for f in features:
        record = {**f['properties']}
        geometries.append(f['geometry'])
        records.append(record)
    
    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")
    return gdf
