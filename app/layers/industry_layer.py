# app/layers/industry_layer.py
"""Industrial area layer for the map."""
import folium
import geopandas as gpd
from branca.colormap import linear


def industry_polygons(industrial_geojson_path: str,
                      name: str = "Industrial Areas"):
    """
    Create a folium layer showing industrial polygon outlines.
    
    Args:
        industrial_geojson_path: Path to industrial polygons GeoJSON
        name: Layer name for folium
        
    Returns:
        folium.GeoJson layer
    """
    try:
        inds = gpd.read_file(industrial_geojson_path).to_crs("EPSG:4326")
    except Exception:
        return None
    
    if inds.empty:
        return None
    
    # Style for industrial areas
    style = {
        "fillColor": "#8B0000",  # Dark red
        "color": "#FF0000",      # Red outline
        "weight": 2,
        "fillOpacity": 0.3
    }
    
    layer = folium.GeoJson(
        inds.__geo_interface__,
        name=name,
        style_function=lambda x: style,
        tooltip=folium.GeoJsonTooltip(
            fields=["landuse"] if "landuse" in inds.columns else [],
            aliases=["Type:"] if "landuse" in inds.columns else []
        )
    )
    return layer


def industry_score_layer(grid_gdf: gpd.GeoDataFrame,
                         key: str = "industry_score",
                         radius: int = 8,
                         opacity: float = 0.6):
    """
    Create a layer showing distance-to-industry score.
    
    Higher score = farther from industry = better.
    
    Args:
        grid_gdf: GeoDataFrame with grid cells
        key: Column name for industry score
        radius: Circle marker radius
        opacity: Fill opacity
        
    Returns:
        List of folium.CircleMarker objects
    """
    markers = []
    
    # Try alternative column names
    if key not in grid_gdf.columns:
        if "dist_to_industry_m" in grid_gdf.columns:
            key = "dist_to_industry_m"
        elif "dist_to_industry" in grid_gdf.columns:
            key = "dist_to_industry"
        else:
            return markers
    
    gdf = grid_gdf.to_crs("EPSG:4326")
    
    # Create colormap - red (close) to green (far)
    vmin = float(gdf[key].min())
    vmax = float(gdf[key].max())
    cmap = linear.RdYlGn_11.scale(vmin, vmax)  # Red to Green
    
    for _, row in gdf.iterrows():
        try:
            centroid = row.geometry.centroid
            val = float(row[key])
            color = cmap(val)
            
            tooltip_text = f"Dist to industry: {val:.0f}m" if "dist" in key else f"Industry score: {val:.1f}"
            
            marker = folium.CircleMarker(
                location=[centroid.y, centroid.x],
                radius=radius,
                color=color,
                fill=True,
                fill_opacity=opacity,
                stroke=False,
                popup=tooltip_text,
                tooltip=tooltip_text
            )
            markers.append(marker)
        except Exception:
            continue
    
    return markers
