# file: etl/fetch_aq_async.py
import asyncio
import aiohttp
import aiofiles
import json
import math
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from scipy.spatial import cKDTree

RAW_DIR = Path("data/raw/air_quality")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ----- Utility: async write JSON -----
async def save_raw_json(name: str, data: dict):
    timestamp = int(time.time())
    out = RAW_DIR / f"{name}_{timestamp}.json"
    async with aiofiles.open(out, "w") as f:
        await f.write(json.dumps(data, ensure_ascii=False, default=str))

# ----- Async HTTP client with retries -----
class AsyncFetcher:
    def __init__(self, concurrency: int = 8, timeout: int = 10):
        self.semaphore = asyncio.Semaphore(concurrency)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    async def fetch_json(self, url: str, params: dict = None, headers: dict = None, name: str = "resp", retries: int = 3, backoff_base: float = 1.0) -> Optional[dict]:
        async with self.semaphore:
            backoff = backoff_base
            for attempt in range(retries):
                try:
                    async with self.session.get(url, params=params, headers=headers) as resp:
                        text = await resp.text()
                        if resp.status == 200:
                            try:
                                data = json.loads(text)
                            except Exception:
                                # fallback: return raw text as dict
                                data = {"raw_text": text}
                            # schedule raw save but don't block caller
                            try:
                                await save_raw_json(name, data)
                            except Exception:
                                pass
                            return data
                        elif resp.status in (429, 502, 503, 504):
                            # retryable HTTP codes
                            await asyncio.sleep(backoff + 0.1 * attempt)
                            backoff *= 2
                        else:
                            # non-retryable: save raw and return None
                            try:
                                await save_raw_json(name + f"_status{resp.status}", {"status": resp.status, "text": text})
                            except Exception:
                                pass
                            return None
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # network or JSON decode error -> retry
                    await asyncio.sleep(backoff)
                    backoff *= 2
            return None

# ----- Fetch open-meteo per coordinate (grid API) -----
async def fetch_open_meteo_for_point(fetcher: AsyncFetcher, lat: float, lon: float) -> Optional[dict]:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,european_aqi"
    }
    key = f"openmeteo_{lat:.5f}_{lon:.5f}"
    data = await fetcher.fetch_json(url, params=params, name=key)
    if not data or "hourly" not in data:
        return None
    hourly = data.get("hourly", {})
    # helper to pick last non-null value
    def last_val(arr):
        if not arr:
            return None
        for v in reversed(arr):
            if v is not None:
                return v
        return None
    res = {"lat": lat, "lon": lon, "source": "open-meteo"}
    mapping = {
        "pm2_5": "pm2_5",
        "pm10": "pm10",
        "carbon_monoxide": "co",
        "nitrogen_dioxide": "no2",
        "sulphur_dioxide": "so2",
        "ozone": "o3",
        "european_aqi": "aqi"
    }
    for k, out_k in mapping.items():
        res[out_k] = last_val(hourly.get(k))
    res["timestamp"] = last_val(hourly.get("time")) if "time" in hourly else None
    return res

# ----- Fetch OpenAQ nearest sensors for a coordinate (within radius meters) -----
async def fetch_openaq_near_point(fetcher: AsyncFetcher, lat: float, lon: float, radius: int = 10000) -> Optional[List[dict]]:
    url = "https://api.openaq.org/v2/latest"
    params = {
        "coordinates": f"{lat},{lon}",
        "radius": radius,
        "limit": 100
    }
    key = f"openaq_{lat:.5f}_{lon:.5f}"
    data = await fetcher.fetch_json(url, params=params, name=key)
    if not data or "results" not in data:
        return None
    parsed = []
    for r in data.get("results", []):
        coords = r.get("coordinates") or {}
        if not coords:
            continue
        base = {"lat": coords.get("latitude"), "lon": coords.get("longitude"), "location": r.get("location")}
        for m in r.get("measurements", []):
            p = m.get("parameter")
            val = m.get("value")
            unit = m.get("unit")
            parsed.append({**base, "parameter": p, "value": val, "unit": unit})
    return parsed

# ----- Convert list of dicts to GeoDataFrame -----
def sensors_to_gdf(sensor_list: List[Dict[str, Any]], crs="EPSG:4326") -> gpd.GeoDataFrame:
    if not sensor_list:
        return gpd.GeoDataFrame(columns=["parameter","value","unit","location","geometry"], crs=crs)
    df = pd.DataFrame(sensor_list)
    df = df.dropna(subset=["lat", "lon"])
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=crs)
    return gdf

# ----- IDW interpolation across sensors -----
def idw_interpolate(grid_gdf: gpd.GeoDataFrame, sensors_gdf: gpd.GeoDataFrame, param: str = "pm25", power: float = 2.0, k: int = 6) -> pd.Series:
    centroids = grid_gdf.geometry.centroid.to_crs(epsg=3857)
    cent_coords = np.vstack([centroids.x.values, centroids.y.values]).T
    s = sensors_gdf[sensors_gdf["parameter"] == param].copy()
    if s.empty:
        return pd.Series([np.nan] * len(grid_gdf), index=grid_gdf.index, name=param)
    s = s.to_crs(epsg=3857)
    sensor_coords = np.vstack([s.geometry.x.values, s.geometry.y.values]).T
    sensor_values = s["value"].values
    tree = cKDTree(sensor_coords)
    kk = min(k, len(sensor_coords))
    dists, idxs = tree.query(cent_coords, k=kk)
    if dists.ndim == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]
    weights = 1.0 / (dists ** power + 1e-12)
    weighted_vals = np.sum(weights * sensor_values[idxs], axis=1) / np.sum(weights, axis=1)
    return pd.Series(weighted_vals, index=grid_gdf.index, name=param)

# ----- Nearest sensor assignment using KDTree (single nearest) -----
def assign_nearest_sensor(grid_gdf: gpd.GeoDataFrame, sensors_gdf: gpd.GeoDataFrame, param: str) -> pd.Series:
    centroids = grid_gdf.geometry.centroid.to_crs(epsg=3857)
    cent_coords = np.vstack([centroids.x.values, centroids.y.values]).T
    s = sensors_gdf[sensors_gdf["parameter"] == param].copy()
    if s.empty:
        return pd.Series([np.nan] * len(grid_gdf), index=grid_gdf.index, name=param)
    s = s.to_crs(epsg=3857)
    sensor_coords = np.vstack([s.geometry.x.values, s.geometry.y.values]).T
    sensor_values = s["value"].values
    tree = cKDTree(sensor_coords)
    _, idxs = tree.query(cent_coords, k=1)
    vals = sensor_values[idxs]
    return pd.Series(vals, index=grid_gdf.index, name=param)

# ----- Sanitize and clip values, compute confidence -----
def sanitize_and_flag(df: pd.DataFrame, col: str, max_reasonable: float = 2000.0) -> Tuple[pd.Series, pd.Series]:
    vals = pd.to_numeric(df[col], errors="coerce")
    # clip unrealistic
    vals = vals.where((vals >= 0) & (vals <= max_reasonable), other=np.nan)
    # confidence heuristics: present -> 1, NaN -> 0
    confidence = (~vals.isna()).astype(float)
    return vals, confidence

# ----- Orchestrator ----- 
async def fetch_open_meteo_and_openaq(fetcher: AsyncFetcher, lat: float, lon: float, cell_id: int) -> dict:
    tasks = [
        fetch_open_meteo_for_point(fetcher, lat, lon),
        fetch_openaq_near_point(fetcher, lat, lon, radius=10000)
    ]
    om, openaq = await asyncio.gather(*tasks)
    return {"cell_id": cell_id, "lat": lat, "lon": lon, "open_meteo": om, "openaq": openaq}

async def fetch_pollution_for_grid(grid_path: str, out_parquet: str = "data/processed/pollution_by_grid.parquet"):
    grid = gpd.read_file(grid_path)
    if "cell_id" not in grid.columns:
        # assume index is the id if absent; create a cell_id column
        grid = grid.reset_index(drop=False).rename(columns={"index": "cell_id"})
    # ensure centroid geometry
    grid["centroid"] = grid.geometry.centroid
    centroids = [(float(pt.y), float(pt.x), int(cid)) for cid, pt in zip(grid["cell_id"], grid["centroid"])]

    fetcher = AsyncFetcher(concurrency=12, timeout=12)
    try:
        tasks = [fetch_open_meteo_and_openaq(fetcher, lat, lon, cid) for lat, lon, cid in centroids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        await fetcher.close()

    per_cell_rows = []
    sensor_rows = []
    for r in results:
        if isinstance(r, Exception) or r is None:
            continue
        cell_id = r.get("cell_id")
        om = r.get("open_meteo") or {}
        row = {"cell_id": cell_id, "lat": r.get("lat"), "lon": r.get("lon")}
        # normalize keys
        row["pm25"] = om.get("pm2_5")
        row["pm10"] = om.get("pm10")
        row["aqi"] = om.get("aqi")
        row["source"] = "open-meteo" if om else None
        per_cell_rows.append(row)
        for s in (r.get("openaq") or []):
            sensor_rows.append(s)

    per_cell_df = pd.DataFrame(per_cell_rows)
    sensor_gdf = sensors_to_gdf(sensor_rows)

    # Merge strategies
    # Work on a copy of grid to avoid mutating caller data
    working = grid.copy().reset_index(drop=True)

    for target_param in [("pm25", "pm25"), ("pm10", "pm10")]:
        out_col = target_param[0]
        # prefer sensors if available -> IDW
        if not sensor_gdf.empty and out_col in sensor_gdf["parameter"].unique():
            interp = idw_interpolate(working, sensor_gdf, param=out_col, power=2.0, k=6)
            working[out_col] = interp.values
            source_col = f"{out_col}_source"
            working[source_col] = "idw-openaq"
        else:
            # fallback to open-meteo per_cell_df by matching cell_id
            if not per_cell_df.empty and out_col in per_cell_df.columns:
                # align by cell_id
                lookup = per_cell_df.set_index("cell_id")[out_col].reindex(working["cell_id"]).values
                working[out_col] = lookup
                working[f"{out_col}_source"] = "open-meteo"
            else:
                working[out_col] = np.nan
                working[f"{out_col}_source"] = None

    # sanitize and compute simple confidence
    for col in ["pm25", "pm10"]:
        vals, conf = sanitize_and_flag(working, col, max_reasonable=2000.0)
        working[col] = vals
        working[f"{col}_confidence"] = conf

    # timestamp and provenance
    ts = int(time.time())
    working["timestamp"] = ts
    # save outputs
    pollution_geo = working[["cell_id", "geometry", "pm25", "pm10", "pm25_source", "pm10_source", "pm25_confidence", "pm10_confidence", "timestamp"]].copy()
    pollution_geo.to_file(PROCESSED_DIR / "pollution_grid.geojson", driver="GeoJSON")
    df_tab = pollution_geo.drop(columns="geometry").copy()
    df_tab["timestamp"] = ts
    df_tab.to_parquet(out_parquet, index=False)
    return df_tab

def run_fetch_pollution(grid_path: str = "data/processed/city_grid.geojson"):
    asyncio.run(fetch_pollution_for_grid(grid_path))