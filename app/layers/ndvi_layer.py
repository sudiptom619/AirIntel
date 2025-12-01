# app/layers/ndvi_layer.py
"""NDVI/greencover layer for the map."""
import folium
import geopandas as gpd
from branca.colormap import linear


def ndvi_choropleth(grid_geojson_path: str,
                    key: str = "ndvi_mean",
                    name: str = "Vegetation (NDVI)"):
    """
    Create a folium choropleth layer for NDVI (vegetation).
    
    Args:
        grid_geojson_path: Path to grid GeoJSON file
        key: Column name for NDVI values
        name: Layer name for folium
        
    Returns:
        folium.Choropleth layer
    """
    g = gpd.read_file(grid_geojson_path).to_crs("EPSG:4326")
    
    if key not in g.columns:
        if "ndvi_norm" in g.columns:
            key = "ndvi_norm"
        elif "green_score" in g.columns:
            key = "green_score"
        else:
            raise ValueError(f"Column {key} not found in grid")
    
    # Prepare minimal geojson
    g_min = g[["geometry", key]].reset_index()
    gj = g_min.__geo_interface__
    
    # Build choropleth - green colorscale for vegetation
    layer = folium.Choropleth(
        geo_data=gj,
        name=name,
        data=g_min,
        columns=["index", key],
        key_on="feature.properties.index",
        fill_color="YlGn",  # Yellow to Green for vegetation
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"{name}"
    )
    return layer


def ndvi_heatmap_layer(grid_gdf: gpd.GeoDataFrame,
                       key: str = "ndvi_mean",
                       radius: int = 8,
                       opacity: float = 0.6):
    """
    Create a heatmap-style NDVI layer using circle markers.
    
    Higher NDVI = greener = better vegetation.
    
    Args:
        grid_gdf: GeoDataFrame with grid cells
        key: Column name for NDVI values
        radius: Circle marker radius
        opacity: Fill opacity
        
    Returns:
        List of folium.CircleMarker objects
    """
    markers = []
    
    if key not in grid_gdf.columns:
        return markers
    
    gdf = grid_gdf.to_crs("EPSG:4326")
    
    # Create colormap - brown to green
    vmin = float(gdf[key].min())
    vmax = float(gdf[key].max())
    cmap = linear.BrBG_11.scale(vmin, vmax)  # Brown to Blue-Green
    
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
                popup=f"NDVI: {val:.3f}",
                tooltip=f"Vegetation: {val:.3f}"
            )
            markers.append(marker)
        except Exception:
            continue
    
    return markers
