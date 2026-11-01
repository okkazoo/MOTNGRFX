"""CPU motion-graphic renderer (numpy / PIL / cv2). Entry point: ``render_frames``."""
from .compositor import Renderer, render_frames
from .text import FONTS

__all__ = ["render_frames", "Renderer", "FONTS"]
