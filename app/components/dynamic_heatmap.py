# app/components/dynamic_heatmap.py
"""Dynamic heatmap generation based on selected location."""
import folium
import geopandas as gpd
import pandas as pd
import numpy as np
from branca.colormap import linear
import math


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two points."""
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(min(1, math.sqrt(a)))
    R = 6371.0
    return R * c


def load_all_predictions(path: str = "data/models/predictions.geojson"):
    """Load all predictions from geojson file without downsampling."""
    try:
        preds = gpd.read_file(path)
        
        if "centroid_lat" in preds.columns and "centroid_lon" in preds.columns:
            df = pd.DataFrame({
                "lat": preds["centroid_lat"].astype(float),
                "lon": preds["centroid_lon"].astype(float),
                "score": preds.get("predicted", preds.get("livability_score", 0)).astype(float),
            })
        else:
            c = preds.geometry.centroid
            df = pd.DataFrame({
                "lat": c.y,
                "lon": c.x,
                "score": preds.get("predicted", preds.get("livability_score", 0)).astype(float)
            })
        
        df = df.dropna(subset=["score"])
        return df
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return None


def filter_heatmap_by_location(heat_df, center_lat, center_lon, radius_km=5.0):
    """
    Filter heatmap points to only show data around selected location.
    
    Args:
        heat_df: DataFrame with lat, lon, score columns
        center_lat: Center latitude
        center_lon: Center longitude
        radius_km: Radius around center to include (default 5km)
    
    Returns:
        Filtered DataFrame
    """
    if heat_df is None or heat_df.empty:
        return heat_df
    
    try:
        mask = heat_df.apply(
            lambda row: haversine_km(
                center_lat, center_lon, 
                float(row["lat"]), float(row["lon"])
            ) <= radius_km,
            axis=1
        )
        return heat_df[mask]
    except Exception as e:
        print(f"Error filtering heatmap: {e}")
        return heat_df


def build_dynamic_heatmap(
    heat_df,
    center_lat,
    center_lon,
    zoom_level=13,
    heat_opacity=0.6,
    heat_radius=6,
    show_entire_dataset=False
):
    """
    Build a folium map with dynamic heatmap centered on selected location.
    
    Args:
        heat_df: DataFrame with lat, lon, score
        center_lat: Center latitude for map
        center_lon: Center longitude for map
        zoom_level: Zoom level for map
        heat_opacity: Opacity of heatmap markers (0-1)
        heat_radius: Radius of circular markers
        show_entire_dataset: If True, show all data; if False, show only nearby
    
    Returns:
        folium.Map object with heatmap
    """
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_level,
        tiles="OpenStreetMap"
    )
    
    if heat_df is None or heat_df.empty:
        return m
    
    try:
        # Get color map
        vmin = float(heat_df["score"].min())
        vmax = float(heat_df["score"].max())
        cmap = linear.YlOrRd_09.scale(vmin, vmax)
        cmap.caption = "Predicted Livability Score"
        
        # Add markers
        for _, row in heat_df.iterrows():
            try:
                score = float(row["score"])
                if not (np.isnan(score) or np.isinf(score)):
                    color = cmap(score)
                    folium.CircleMarker(
                        location=[row["lat"], row["lon"]],
                        radius=heat_radius,
                        color=color,
                        fill=True,
                        fill_opacity=heat_opacity,
                        stroke=False,
                        popup=f"Score: {score:.1f}",
                        tooltip=f"Score: {score:.1f}"
                    ).add_to(m)
            except Exception:
                continue
        
        m.add_child(cmap)
        
        # Add dataset info to map
        info_text = f"Showing {len(heat_df)} locations"
        if not show_entire_dataset:
            info_text += f" within 5km of selected location"
        
    except Exception as e:
        print(f"Error building heatmap: {e}")
    
    return m


def add_selected_marker(m, lat, lon, label="Selected Location", color="blue"):
    """
    Add a marker for the selected location.
    
    Args:
        m: folium.Map object
        lat: Latitude
        lon: Longitude
        label: Marker label
        color: Marker color (blue, red, green, etc)
    
    Returns:
        Updated folium.Map
    """
    folium.CircleMarker(
        location=[lat, lon],
        radius=12,
        color=f"#{color if color.startswith('#') else '0000ff' if color == 'blue' else 'ff0000'}",
        fill=True,
        fill_color=f"#{color if color.startswith('#') else '0000ff' if color == 'blue' else 'ff0000'}",
        fill_opacity=0.9,
        popup=label,
        tooltip=label,
        weight=3
    ).add_to(m)
    
    return m


def add_radius_circle(m, lat, lon, radius_km=5.0, label="Search Radius"):
    """
    Add a circle showing the search radius.
    
    Args:
        m: folium.Map object
        lat: Center latitude
        lon: Center longitude
        radius_km: Radius in kilometers
        label: Circle label
    
    Returns:
        Updated folium.Map
    """
    # Convert km to meters (folium uses meters)
    radius_m = radius_km * 1000
    
    folium.Circle(
        location=[lat, lon],
        radius=radius_m,
        color="gray",
        fill=False,
        weight=2,
        opacity=0.5,
        popup=label,
        tooltip=label
    ).add_to(m)
    
    return m


def add_two_location_markers(m, lat_a, lon_a, lat_b, lon_b):
    """
    Add markers and line for two locations on comparison map.
    
    Args:
        m: folium.Map object
        lat_a, lon_a: Location A coordinates
        lat_b, lon_b: Location B coordinates
    
    Returns:
        Updated folium.Map with both markers and connecting line
    """
    # Location A marker (blue)
    folium.CircleMarker(
        location=[lat_a, lon_a],
        radius=12,
        color="#0000ff",
        fill=True,
        fill_color="#0000ff",
        fill_opacity=0.9,
        popup="📍 Location A",
        tooltip="📍 Location A",
        weight=3
    ).add_to(m)
    
    # Location B marker (red)
    folium.CircleMarker(
        location=[lat_b, lon_b],
        radius=12,
        color="#ff0000",
        fill=True,
        fill_color="#ff0000",
        fill_opacity=0.9,
        popup="📍 Location B",
        tooltip="📍 Location B",
        weight=3
    ).add_to(m)
    
    # Connecting line
    folium.PolyLine(
        locations=[[lat_a, lon_a], [lat_b, lon_b]],
        color="gray",
        weight=2,
        opacity=0.6,
        popup="Distance between locations"
    ).add_to(m)
    
    return m