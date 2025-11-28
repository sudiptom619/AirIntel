from features.engineer_features import compute_ndvi_features, folium_ndvi_map
g = compute_ndvi_features()
m = folium_ndvi_map(g)
m  # Jupyter will render the folium map