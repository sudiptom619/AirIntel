# scripts/smoke_phase1.py
import sys
import os

ROOT = os.path.dirname(os.path.dirname(__file__))  # goes up from /scripts/
sys.path.append(ROOT)

from etl.utils.geo import bbox_from_center, create_square_grid
from etl.utils.api_clients import SimpleRequestClient

def smoke_test():
    # test grid creation
    bbox = bbox_from_center(22.5726, 88.3639, delta_deg=0.02)  # Kolkata small bbox
    grid = create_square_grid(bbox, resolution_m=1000)
    print("Grid cells:", len(grid))
    # test HTTP wrapper with a simple call (example using Nominatim)
    client = SimpleRequestClient()
    res = client.get("https://nominatim.openstreetmap.org/search", params={"q":"Kolkata", "format":"json", "limit":1})
    print("Nominatim search first result keys:", list(res[0].keys()) if res else "no result")

if __name__ == "__main__":
    smoke_test()
