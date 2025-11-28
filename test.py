import folium
import geopandas as gpd
grid = gpd.read_file("data/processed/city_grid_with_osm.geojson")
m = folium.Map(location=[grid.centroid.y.mean(), grid.centroid.x.mean()], zoom_start=12)
# add choropleth for road_density
folium.Choropleth(geo_data=grid.to_json(), data=grid, key_on="feature.properties.cell_id",
                  columns=["cell_id","road_density_m_per_m2"], fill_opacity=0.7, line_opacity=0.2).add_to(m)
m.save("debug_osm_roads.html")
