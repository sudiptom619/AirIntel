# app/layers/flood_layer.py
"""Flood risk layer for the map."""
import folium
import geopandas as gpd
from branca.colormap import linear


def flood_choropleth(grid_geojson_path: str,
                     key: str = "flood_proxy",
                     name: str = "Flood Risk"):
    """
    Create a folium choropleth layer for flood risk.
    
    Args:
        grid_geojson_path: Path to grid GeoJSON file
        key: Column name for flood risk values
        name: Layer name for folium
        
    Returns:
        folium.Choropleth layer
    """
    g = gpd.read_file(grid_geojson_path).to_crs("EPSG:4326")
    
    if key not in g.columns:
        if "flood_score" in g.columns:
            key = "flood_score"
        else:
            raise ValueError(f"Column {key} not found in grid")
    
    # Prepare minimal geojson
    g_min = g[["geometry", key]].reset_index()
    gj = g_min.__geo_interface__
    
    # Build choropleth - blue for flood risk (higher = more risk)
    layer = folium.Choropleth(
        geo_data=gj,
        name=name,
        data=g_min,
        columns=["index", key],
        key_on="feature.properties.index",
        fill_color="Blues",  # Light to dark blue for flood
        fill_opacity=0.5,
        line_opacity=0.2,
        legend_name=f"{name} (0-1)"
    )
    return layer


def flood_heatmap_layer(grid_gdf: gpd.GeoDataFrame,
                        key: str = "flood_proxy",
                        radius: int = 8,
                        opacity: float = 0.5):
    """
    Create a heatmap-style flood risk layer using circle markers.
    
    Higher values = higher flood risk.
    
    Args:
        grid_gdf: GeoDataFrame with grid cells
        key: Column name for flood risk
        radius: Circle marker radius
        opacity: Fill opacity
        
    Returns:
        List of folium.CircleMarker objects
    """
    markers = []
    
    if key not in grid_gdf.columns:
        return markers
    
    gdf = grid_gdf.to_crs("EPSG:4326")
    
    # Create colormap - light blue to dark blue
    vmin = float(gdf[key].min())
    vmax = float(gdf[key].max())
    cmap = linear.Blues_09.scale(vmin, vmax)
    
    for _, row in gdf.iterrows():
        try:
            centroid = row.geometry.centroid
            val = float(row[key])
            color = cmap(val)
            
            risk_label = "Low" if val < 0.3 else ("Medium" if val < 0.7 else "High")
            
            marker = folium.CircleMarker(
                location=[centroid.y, centroid.x],
                radius=radius,
                color=color,
                fill=True,
                fill_opacity=opacity,
                stroke=False,
                popup=f"Flood risk: {val:.2f} ({risk_label})",
                tooltip=f"Flood: {risk_label}"
            )
            markers.append(marker)
        except Exception:
            continue
    
    return markers
