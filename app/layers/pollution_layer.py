# app/layers/pollution_layer.py
import folium
import geopandas as gpd

def pollution_choropleth(grid_geojson_path, key="pm25"):
    g = gpd.read_file(grid_geojson_path).to_crs("EPSG:4326")
    # prepare minimal geojson to speed up
    g_min = g[["geometry", key]].reset_index()
    gj = g_min.__geo_interface__
    # build choropleth (you can control bins or continuous colormap)
    layer = folium.Choropleth(
        geo_data=gj,
        name=f"Pollution ({key})",
        data=g_min,
        columns=["index", key],
        key_on="feature.properties.index",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"{key} value"
    )
    return layer
