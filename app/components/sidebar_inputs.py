"""Small sidebar helpers for the Streamlit app.

This module contains tiny helpers used by `app/streamlit_app.py` so the
sidebar logic can be reused or tested separately.
"""
from geopy.geocoders import Nominatim


def geocode(query: str, user_agent: str = "livability_app"):
    """Return (lat, lon, address) for a text query using Nominatim.

    Returns (None, None, None) when no match found or on error.
    """
    if not query:
        return None, None, None

    try:
        geocoder = Nominatim(user_agent=user_agent)
        loc = geocoder.geocode(query)
        if not loc:
            return None, None, None
        return loc.latitude, loc.longitude, loc.address
    except Exception:
        # keep sidebar robust — don't raise here
        return None, None, None
