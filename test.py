import geopandas as gpd
grid = gpd.read_file("data/processed/city_grid.geojson")
bbox = grid.total_bounds   # [minx, miny, maxx, maxy]
print(bbox)
