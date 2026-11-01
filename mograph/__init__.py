"""ComfyUI-Mograph core: pure-Python analysis, planning, audio and rendering.

No ComfyUI or torch imports live under this package; it runs standalone on CPU.
Submodules are imported explicitly (``from mograph import plan``) so a missing
optional piece never breaks the others.
"""

__version__ = "0.1.0"
