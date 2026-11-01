"""lines: the module's own strand form-in / form-out (what every phase-1/2 module did). Default style: with it a module draws
exactly what it drew before the I/O split."""
from mograph.modules.io.base import IOStyle, register_style


@register_style("lines")
class Lines(IOStyle):
    description = ("The module's own form and exit: strands arrive along the entry direction, draw the type, and on exit "
                   "run away along the exit direction. The default; hands strands to a transition unchanged.")
    is_lines = True
    material_kind = "strands"

    def layers(self, t, p):
        return self.body.raw_layers(t)

    def material(self):
        from mograph.modules.material import Material
        return Material.from_bundle(self.body._rawcall("entry_bundle" if self.side == "in" else "exit_bundle"))

    def focus(self, t, p):
        return self.body._rawcall("focus", t)
