"""P20 Delta (v005): 2 shots, small precise type woven into river contours, silver black-and-white. 90 bpm (tracked 89.2); downbeats 3.266 / 5.956 / 8.646 / 11.336 / 14.026.
Beats follow 15_fusion/v005/DESIGNS.md "P20 Delta" (EVERY RIVER 2.667 / BENDS 5.333 / AND BRAIDS 8.0 / TOWARD 10.667 / THE SEA 13.333), shots + grid from MEDIA.md (cut 8.646).
Lego: edge_run_small_type (small_line layout, cap height >= 4 % of the frame), contour_hand_off, matte_edge_reveal_wipe, chain_through_cards (tilt_to_plan then pullback_reveal).
    python 15_fusion/pieces/_gen_P20.py [--check]"""
import _v005_lib as L
from _v005_lib import port
from tools.fusion import edge_run, plan_rules

plan_rules.SETTLE_HOLD = 1.0      # v005 QC: the river flow smeared the small type at the holds (settle 0.75 leaves 25 % of the footage flow in the type)

DESIGN = dict(
    id="P20", slug="delta", bpm_design=90.0, anchor=(2.6667, 3.266), pace="flowing", palette="silver_bw", type_role="small_precise",
    camera_lead="tilt_to_plan", technique="built_in", mood="calm", seed=20, cards=4,
    post=[edge_run.weave, L.legible_camera],
    blocks=["edge_run_small_type", "contour_hand_off", "matte_edge_reveal_wipe", "chain_through_cards", "locked_plate_move"],
    scene="a braided river delta seen from the air in black and white, silver channels, mist, the river meeting the sea",
    why="v005 design P20: small precise type woven along the river contours, two aerial shots, tilt_to_plan then pullback_reveal (DESIGNS.md)",
    deferred=["v005 round 2 (legibility): small_line is the edge_run block's layout (letters >= 4 %), so only BENDS keeps it; EVERY RIVER (EVERY / RIVER) and AND BRAIDS (AND / BRAIDS, solid_hollow depth 1.2 + grey walls: the vermilion slab walls hid the white caps; in a full layout the solid line is 590 px tall and unreadable) take hung_from_top, AND BRAIDS in Passion One (the pixel blackletter turned the slab caps into blobs), TOWARD takes full (pullback_reveal fill 0.58), THE SEA edge_L -> full (THE / SEA: spectrum draws the word at half the box, 6 % letters), TOWARD outflow tilt 0.3 / 5 rings (the ghost rows filled the frame): every other hold needs letters >= 7 % of the frame (45 px at 640), the small line gave 5.5 %, the bottom-right quadrant 4.7 %",
              "P20 mattes are depth thresholds (MEDIA.md): on a top-down shot that is the lower third, not a subject; matte treatments act on it as is",],
    words=[
        dict(text="EVERY RIVER", lines=["EVERY", "RIVER"], font="Rubik Lines", module="topo_tower", hit=2.667, hold=2.0, exit_s=0.6, gap=0.5, io=("lines", "lines"), layout="hung_from_top",
             entry=port("iso", "up"), via=port("edge", "along"), transition="circuit_board", slot=2, camera="tilt_to_plan"),
        dict(text="BENDS", font="Geist Pixel", module="circuit", hit=5.333, hold=1.8, exit_s=0.5, gap=0.5, io=("lines", "lines"), layout="small_line",
             entry=port("edge", "along"), matte="glow", via=port("matte", "around"), transition="spiral_around", slot=1, camera="tilt_to_plan",
             edge_run=dict(source="luma_edges", cap_frac=0.05)),
        dict(text="AND BRAIDS", lines=["AND", "BRAIDS"], font="Passion One", module="solid_hollow", hit=8.0, hold=2.0, exit_s=0.5, gap=0.5, io=("lines", "lines"), layout="hung_from_top",
             entry=port("edge", "up"), matte="wipe", via=port("edge", "along"), transition="braid", slot=1, camera="pullback_reveal", yaw=6, params=dict(depth=1.2, band=(70, 70, 78))),
        dict(text="TOWARD", font="Codystar", module="outflow", hit=10.667, hold=1.7, exit_s=0.5, gap=0.5, io=("lines", "lines"), layout="full",
             entry=port("edge", "up"), via=port("iso", "along"), transition="ripple", slot=1, camera=("pullback_reveal", dict(fill=0.58)), yaw=6, params=dict(tilt=0.3, rings=5)),
        dict(text="THE SEA", lines=["THE", "SEA"], font="Space Grotesk", module="spectrum", hit=13.333, hold=2.0, exit_s=0.5, io=("lines", "particles"), layout="full",
             exit=port("edge", "up"), slot=1, camera=("pullback_reveal", dict(fill=0.58)), yaw=6),
    ],
    shots=[dict(id="s1", still="A", frames=209, span=207, seed=201,
                prompt="the drone camera slowly tilts down in one continuous move from a low oblique view until it looks straight down on the braided river "
                       "channels, water glints, mist drifts, black and white"),
           dict(id="s2", still="B", frames=185, span=177, seed=202,
                prompt="the drone camera looks straight down and slowly rises and pulls back in one continuous move, revealing the whole river bend meeting "
                       "the sea, waves roll in, black and white")],
    stills=[dict(id="A", seed=2001, neg=", colour", prompt="black and white aerial photograph at a low oblique angle over a braided river, silver channels splitting and "
                                                           "rejoining across grey gravel flats, mist banks, soft morning light, vertical frame"),
            dict(id="B", seed=2002, neg=", colour", prompt="black and white aerial photograph looking down on a wide river bend where it meets the sea, waves rolling "
                                                           "into the river mouth, vertical frame")],
)

if __name__ == "__main__":
    L.main(DESIGN)
