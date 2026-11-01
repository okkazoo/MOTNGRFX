"""mograph.modules: lego type modules (word modules, transition modules, sequencer). See README.md."""
from .core import (Port, Placement, Timing, Bundle, Pairing, adapt, Palette, WordModule, TransitionModule, register,
                   list_modules, get_module)
from .sequence import Sequence
from . import words, transitions  # noqa: F401  (importing registers the built-in modules)
from .material import Material  # noqa: F401  (phase 3a: typed hand-off)
from . import io  # noqa: F401  (phase 3a: form-in / form-out styles)
