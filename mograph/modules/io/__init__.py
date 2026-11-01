"""mograph.modules.io: form-in / form-out styles (phase 3a). A word module is a body (its look in the hold); how it materialises and
dissolves is a style: 'lines' (default, the module's own strands), 'blobs', 'flicker', 'particles', 'wipe'.
Use: WordModule(..., form_in='blobs', form_out=('flicker', {'beat': 0.5}))."""
from .base import IOStyle, register_style, get_style, list_styles, STYLES  # noqa: F401
from . import lines, wipe, flicker, outline  # noqa: F401   outline: 2026-11-01 glyph-outline hand-off of the library transitions
for _m in ("blobs", "particles"):
    try:
        __import__("mograph.modules.io." + _m)
    except ModuleNotFoundError as e:
        if _m not in str(e):
            raise
