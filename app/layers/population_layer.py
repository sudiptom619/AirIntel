# app/layers/population_layer.py
"""Population density layer for the map."""
import folium
import geopandas as gpd
from branca.colormap import linear


def population_choropleth(grid_geojson_path: str,
                          key: str = "pop_density",
                          name: str = "Population Density"):
    """
    Create a folium choropleth layer for population density.
    
    Args:
        grid_geojson_path: Path to grid GeoJSON file
        key: Column name for population density values
        name: Layer name for folium
        
    Returns:
        folium.Choropleth layer
    """
    g = gpd.read_file(grid_geojson_path).to_crs("EPSG:4326")
    
    if key not in g.columns:
        if "pop_score" in g.columns:
            key = "pop_score"
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
        fill_color="BuPu",  # Blue-Purple for population
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f"{name} (people/km²)"
    )
    return layer


def population_heatmap_layer(grid_gdf: gpd.GeoDataFrame,
                             key: str = "pop_density",
                             radius: int = 8,
                             opacity: float = 0.6):
    """
    Create a heatmap-style population layer using circle markers.
    
    Args:
        grid_gdf: GeoDataFrame with grid cells
        key: Column name for population density
        radius: Circle marker radius
        opacity: Fill opacity
        
    Returns:
        List of folium.CircleMarker objects
    """
    markers = []
    
    if key not in grid_gdf.columns:
        return markers
    
    gdf = grid_gdf.to_crs("EPSG:4326")
    
    # Create colormap - light to dark purple
    vmin = float(gdf[key].min())
    vmax = float(gdf[key].max())
    cmap = linear.BuPu_09.scale(vmin, vmax)
    
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
                popup=f"Pop: {val:.0f}/km²",
                tooltip=f"Population: {val:.0f}/km²"
            )
            markers.append(marker)
        except Exception:
            continue
    
    return markers
