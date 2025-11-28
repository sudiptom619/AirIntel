"""
Defensive import for clip_and_reproject. If the symbol isn't present in
etl.utils.geo we show the available names so you can fix/rename/export it.
"""
try:
    from .geo import clip_and_reproject  # expected symbol
except Exception:
    import importlib
    mod = importlib.import_module("etl.utils.geo")
    available = [n for n in dir(mod) if not n.startswith("_")]
    raise ImportError(
        "etl.utils.geo does not define 'clip_and_reproject'. "
        f"Available names in etl.utils.geo: {available}. "
        "Either add/rename the function to 'clip_and_reproject' or export it."
    )

__all__ = ["clip_and_reproject"]