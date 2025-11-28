# app/utils/geo.py
import geopandas as gpd
from shapely.geometry import box
import pyproj

def bbox_from_center(lat, lon, delta_deg=0.05):
    """
    Return bbox in (south, west, north, east) given center and delta degrees.
    Useful quick helper; prefer bbox in meters in production.
    """
    return (lat - delta_deg, lon - delta_deg, lat + delta_deg, lon + delta_deg)

def create_square_grid(bbox, resolution_m=1000, crs_epsg=3857):
    """
    Create a square grid of polygons over bbox (lat/lon in EPSG:4326).
    Steps:
     - create a projected bbox in meters (EPSG:3857) for uniform cell size
     - create square polygons of size resolution_m
     - return GeoDataFrame in EPSG:4326
    """
    # Convert bbox to projected CRS
    bbox_polygon = box(bbox[1], bbox[0], bbox[3], bbox[2])  # lon/lat order
    gdf = gpd.GeoDataFrame({'geometry': [bbox_polygon]}, crs="EPSG:4326")
    gdf_proj = gdf.to_crs(epsg=crs_epsg)
    minx, miny, maxx, maxy = gdf_proj.total_bounds

    xs = list(range(int(minx), int(maxx) + resolution_m, resolution_m))
    ys = list(range(int(miny), int(maxy) + resolution_m, resolution_m))

    cells = []
    for x in xs:
        for y in ys:
            cells.append(box(x, y, x + resolution_m, y + resolution_m))

    grid = gpd.GeoDataFrame({'geometry': cells}, crs=f"EPSG:{crs_epsg}")
    grid = grid.to_crs(epsg=4326)  # back to lat/lon for storage/visualization
    grid["cell_id"] = grid.index.astype(str)
    return grid

def clip_and_reproject(src_path, dst_path, bbox, dst_crs="EPSG:3857"):
    """
    Clip raster at `src_path` to bbox (minx,miny,maxx,maxy) in EPSG:4326
    and reproject to `dst_crs`. Writes result to `dst_path`.
    """
    import rasterio
    from rasterio.mask import mask
    from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
    from shapely.geometry import box
    import numpy as np

    # bbox is expected in EPSG:4326
    src_bbox = bbox

    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError("Source raster has no CRS")

        # transform bbox to source CRS bounds (for cropping)
        src_bounds = transform_bounds("EPSG:4326", src.crs.to_string(),
                                      src_bbox[0], src_bbox[1], src_bbox[2], src_bbox[3],
                                      densify_pts=21)

        # create geometry in source CRS for masking
        geom_wgs = box(*src_bbox).__geo_interface__
        from rasterio.warp import transform_geom
        geom_src = transform_geom("EPSG:4326", src.crs.to_string(), geom_wgs)

        # mask (crop) to geometry in source CRS
        out_image, out_transform = mask(src, [geom_src], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": src.crs
        })

    # calculate transform/size for destination CRS using the source bbox in src CRS
    dst_transform, dst_width, dst_height = calculate_default_transform(
        out_meta["crs"], dst_crs, out_meta["width"], out_meta["height"],
        *src_bounds
    )

    dst_meta = out_meta.copy()
    dst_meta.update({
        "crs": dst_crs,
        "transform": dst_transform,
        "width": int(dst_width),
        "height": int(dst_height)
    })

    # reproject each band
    count = out_image.shape[0]
    dst_array = np.zeros((count, dst_meta["height"], dst_meta["width"]), dtype=out_image.dtype)

    for i in range(count):
        reproject(
            source=out_image[i],
            destination=dst_array[i],
            src_transform=out_transform,
            src_crs=out_meta["crs"],
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest
        )

    # write output
    with rasterio.open(dst_path, "w", **dst_meta) as dst:
        dst.write(dst_array)

    return dst_path
