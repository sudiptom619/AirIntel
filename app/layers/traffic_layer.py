# app/layers/traffic_layer.py
"""Traffic/road density layer for the map."""
import folium
import geopandas as gpd
from branca.colormap import linear


def traffic_choropleth(grid_geojson_path: str, 
                       key: str = "road_density_m_per_m2",
                       name: str = "Traffic Density"):
    """
    Create a folium choropleth layer for road/traffic density.
    
    Args:
        grid_geojson_path: Path to grid GeoJSON file
        key: Column name for traffic density values
        name: Layer name for folium
        
    Returns:
        folium.Choropleth layer
    """
    g = gpd.read_file(grid_geojson_path).to_crs("EPSG:4326")
    
    # Fallback to road_length_m if density not available
    if key not in g.columns:
        if "road_length_m" in g.columns and "area_m2" in g.columns:
            g[key] = g["road_length_m"] / g["area_m2"]
        elif "road_length_m" in g.columns:
            key = "road_length_m"
        else:
            raise ValueError(f"Column {key} not found in grid")
    
    # Prepare minimal geojson
    g_min = g[["geometry", key]].reset_index()
    gj = g_min.__geo_interface__
    
    # Build choropleth
    layer = folium.Choropleth(
        geo_data=gj,
        name=name,
        data=g_min,
        columns=["index", key],
        key_on="feature.properties.index",
        fill_color="YlOrBr",  # Yellow to Brown for traffic
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"{name} (m/m²)"
    )
    return layer


def traffic_heatmap_layer(grid_gdf: gpd.GeoDataFrame,
                          key: str = "road_density_m_per_m2",
                          radius: int = 8,
                          opacity: float = 0.6):
    """
    Create a heatmap-style traffic layer using circle markers.
    
    Args:
        grid_gdf: GeoDataFrame with grid cells
        key: Column name for traffic density values  
        radius: Circle marker radius
        opacity: Fill opacity
        
    Returns:
        List of folium.CircleMarker objects
    """
    markers = []
    
    if key not in grid_gdf.columns:
        return markers
    
    gdf = grid_gdf.to_crs("EPSG:4326")
    
    # Create colormap
    vmin = float(gdf[key].min())
    vmax = float(gdf[key].max())
    cmap = linear.YlOrBr_09.scale(vmin, vmax)
    
    for _, row in gdf.iterrows():
        try:
            centroid = row.geometry.centroid
            val = float(row[key])
            color = cmap(val)
            
            marker = folium.CircleMarker(
                location=[centroid.y, centroid.x],
                radius=radius,
                color=color,
                fill=True,
                fill_opacity=opacity,
                stroke=False,
                popup=f"Traffic: {val:.4f}",
                tooltip=f"Road density: {val:.4f} m/m²"
            )
            markers.append(marker)
        except Exception:
            continue
    
    return markers
