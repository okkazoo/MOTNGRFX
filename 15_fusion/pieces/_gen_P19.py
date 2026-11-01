"""P19 Gather (v005): ONE 16 s orbit, type made of sparks and edges around molten glass, amber / oxblood duotone. 105 bpm (design downbeat 2.286 = tracked 2.187).
Beats follow 15_fusion/v005/DESIGNS.md "P19 Gather" (SHAPE 1.143 / THE FIRE 4.0 / BEFORE 6.857 / IT COOLS 9.714 / INTO GLASS 12.571), shot + grid from MEDIA.md.
Lego: type_from_sparks (BEFORE), fill_silhouette_grow (INTO GLASS), half_front_half_behind, window_ending, contour_hand_off, chain_through_cards (orbit_even).
    python 15_fusion/pieces/_gen_P19.py [--check]"""
import _v005_lib as L
from _v005_lib import port, HOLD_CAM

DESIGN = dict(
    id="P19", slug="gather", bpm_design=105.0, anchor=(2.2857, 2.187), pace="building", palette="duotone_warm", type_role="from_footage",
    camera_lead="orbit_even", technique="behind", mood="hype", seed=19, cards=4,
    blocks=["type_from_sparks", "fill_silhouette_grow", "half_front_half_behind", "contour_hand_off", "chain_through_cards"],
    scene="a glowing molten glass gather on a steel blowpipe, sparks, a furnace glowing behind, black workshop",
    why="v005 design P19: one even orbit, type built from the sparks and edges of the footage, fills the gather silhouette (DESIGNS.md)",
    deferred=["v005 round 2 (legibility): THE FIRE massive_crop -> full (E was cut), BEFORE small_line -> hung_from_top (small_line is the edge_run block's, 4 % letters; the hold needs 7 %), "
              "IT COOLS in two lines IT / COOLS and Silkscreen (Rock 3D is hollow: the cube lattice kept a few cubes; a very heavy face fills the lattice into one block), voxel_orbit depth 20, orbit_even fill 0.58 on the full holds",
              "BEFORE: the spark blobs of the plate as sources of the particles module (block type_from_sparks, missing part): the module draws its own streaks",
              "IT COOLS colour cools amber -> steel blue on the hold (block colour_shift_on_hold is partial: no per-hold tint timeline wired)",
              "INTO GLASS glyphs re-fitted to the live gather silhouette: the fill matte treatment keeps the line inside the matte; the downbeat phase of this grid is a guess (conf 1.05)"],
    words=[
        dict(text="SHAPE", font="Facade", module="pipework", hit=1.143, hold=1.8, exit_s=0.7, gap=0.5, io=("flicker", "blobs"), layout=("quadrant_stack", "bl"),
             entry=port("edge", "up"), matte="front", via=port("matte", "around"), transition="twist", slot=1, camera="orbit_even"),
        dict(text="THE FIRE", lines=["THE", "FIRE"], font="Fascinate", module="lanes", hit=4.0, hold=2.0, exit_s=0.6, gap=0.5, io=("lines", "lines"), layout="full",
             entry=port("edge", "up"), matte="front", via=port("edge", "along"), transition="fan", slot=1, camera=("orbit_even", dict(fill=0.58))),
        dict(text="BEFORE", font="Doto", module="particles", hit=6.857, hold=1.8, exit_s=0.6, gap=0.5, io=("particles", "particles"), layout="hung_from_top",
             entry=port("edge", "up"), matte="window", via=port("iso", "along"), transition="ripple", slot=0, camera="orbit_even"),
        dict(text="IT COOLS", lines=["IT", "COOLS"], font="Silkscreen", module="voxel_orbit", hit=9.714, hold=2.4, exit_s=0.6, gap=0.5, io=("lines", "flicker"), layout="full",
             matte="glow", transition="cut", slot=1, camera=HOLD_CAM, params=dict(depth=20.0)),
        dict(text="INTO GLASS", lines=["INTO", "GLASS"], font="Foldit", module="tangent_spray", hit=12.571, hold=3.0, exit_s=0.8, io=("lines", "particles"), layout="full",
             entry=port("matte", "up"), matte="fill", slot=2, camera=("orbit_even", dict(fill=0.58))),
    ],
    shots=[dict(id="s1", still="A", frames=385, span=384, seed=191,
                prompt="the camera orbits slowly and evenly around the glowing molten glass gather in one continuous move from left to right, the blowpipe turns "
                       "steadily, the gather swells and its surface folds and flows, small sparks rise, the furnace glows behind, heat shimmer, cinematic")],
    stills=[dict(id="A", seed=1901, prompt="macro photograph of a glowing molten glass gather on the end of a steel blowpipe, bright orange core, surface folds, "
                                           "sparks and heat shimmer, furnace mouth glowing deep red behind, black workshop, vertical frame, cinematic")],
)

if __name__ == "__main__":
    L.main(DESIGN)
