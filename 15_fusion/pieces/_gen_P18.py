"""P18 Bowl (v005): 6 rapid downbeat cuts, flat graphic strips / blocks over video, saturated primaries on bleached concrete. 140 bpm half-time.
Beats follow 15_fusion/v005/DESIGNS.md "P18 Bowl" (CARVE 1.714 / CONCRETE 4.286 / FALL 6.857 / GET UP 9.429 / FLY AGAIN 12.0), shots + grid (phase 0) from MEDIA.md.
Lego: strips_persist_across_cuts (the strip of CONCRETE stays up over the cut), matte_edge_reveal_wipe, fill_silhouette_grow (GET UP), contour_hand_off,
half_front_half_behind, window_ending (FLY AGAIN), chain_through_cards.
    python 15_fusion/pieces/_gen_P18.py [--check]"""
import _v005_lib as L
from _v005_lib import port, HOLD_CAM
from tools.fusion import strip_resnap

DESIGN = dict(
    id="P18", slug="bowl", bpm_design=140.0, anchor=(0.0, 0.025), pace="driving", palette="flat_primaries", type_role="flat_strips",
    camera_lead="graze", technique="behind", mood="playful", seed=18, cards=4,
    post=[strip_resnap.weave, L.legible_camera],
    blocks=["strips_persist_across_cuts", "matte_edge_reveal_wipe", "fill_silhouette_grow", "contour_hand_off", "half_front_half_behind"],
    scene="a skateboarder in an empty concrete skate bowl, late sun, bleached concrete, deep blue sky",
    why="v005 design P18: 6 rapid downbeat cuts, flat strips and blocks as the continuity over cutting video, saturated primaries (DESIGNS.md)",
    deferred=["v005 round 2 (legibility): CARVE massive_crop -> hung_from_top (C and E were cut), CONCRETE in two lines CONC / RETE (one line gave 29 px letters on the strip, the rule is 45 px), "
              "FALL voxel_roll in Bitcount (Rubik Glitch Pop has holes: the cube lattice lost the letters) without the grid floor (grid=False, 5 rows, lean 6: the letters were lost in the grid, flat cubes read as a pixel face), GET UP scale_clash reach 0.35 / 12 slabs (the 1.4-wide, 26-slab extrusion "
              "ran across the frame), spin_read_spin fill 0.58",
              "FALL letters in cells (P9) and graze / spin_read_spin along the strips: backing shapes are screen space, so lines with shapes are held (hold_drift); "
              "FALL uses a full layout with spin_read_spin",
              "rise trail 9.4-10.3 on GET UP (masktrack rise): not in the piece (it would regenerate s5 with v2v)",
              "GET tiny in the counter of the P (P4) is the scale_clash module own small word, not a separate placement"],
    words=[
        dict(text="CARVE", font="Erica One", module="gradient_fade", hit=1.714, hold=1.7, exit_s=0.5, gap=0.5, io=("lines", "lines"), layout="hung_from_top",
             entry=port("iso", "up"), matte="front", via=port("edge", "along"), transition="fan", slot=1, camera="graze", yaw=8),
        dict(text="CONCRETE", lines=["CONC", "RETE"], font="Silkscreen", module="beat_bars", hit=4.286, hold=1.7, exit_s=0.5, gap=0.5, io=("lines", "wipe"), layout=("solid_strips", "wonky"),
             entry=port("edge", "up"), matte="wipe", transition="cut", slot=2),
        dict(text="FALL", font="Bitcount", module="voxel_roll", hit=6.857, hold=1.7, exit_s=0.8, gap=0.5, io=("flicker", "lines"), layout="full",
             via=port("edge", "along"), transition="circuit_board", slot=0, camera=("spin_read_spin", dict(fill=0.58)), yaw=6,
             params=dict(grid=False, rows=5, lean=6.0)),
        dict(text="GET UP", font="Passion One", module="scale_clash", hit=9.429, hold=1.7, exit_s=0.8, gap=0.5, io=("lines", "lines"), layout="full",
             entry=port("matte", "up"), matte="fill", via=port("matte", "around"), transition="spiral_around", slot=1,
             camera=("spin_read_spin", dict(fill=0.72)), yaw=6, params=dict(reach=0.35, slabs=12)),
        dict(text="FLY AGAIN", lines=["FLY", "AGAIN"], font="Bungee Outline", module="hollow_tubes", hit=12.0, hold=1.3, exit_s=0.4, io=("lines", "blobs"),
             layout=("split_sides", "type_first"), entry=port("edge", "up"), matte="window", slot=1),
    ],
    shots=[dict(id="s1", still="A", frames=89, span=83, seed=181, prompt="the camera skims low along the concrete coping in one smooth lateral move as the skateboarder carves past"),
           dict(id="s2", still="A", frames=49, span=41, seed=182, prompt="the camera skims low along the coping the other way, wheels kick dust"),
           dict(id="s3", still="C", frames=49, span=41, seed=183, prompt="static locked-off camera, the skateboarder rises in the air above the rim and starts to tip"),
           dict(id="s4", still="B", frames=49, span=41, seed=184, prompt="static locked-off overhead camera, the skateboarder lies still on the bowl floor, board rolling away"),
           dict(id="s5", still="B", frames=49, span=41, seed=185, prompt="static locked-off overhead camera, the skateboarder gets up and stands, picking up his board"),
           dict(id="s6", still="C", frames=89, span=82, seed=186,
                prompt="the camera skims low along the rim in one smooth lateral move as the skateboarder launches high into the blue sky")],
    stills=[dict(id="A", seed=1801, prompt="low angle along the coping of an empty concrete skate bowl, a skateboarder carving the wall, late sun, bleached pale concrete, deep blue sky"),
            dict(id="B", seed=1802, prompt="overhead view straight down into a concrete skate bowl, a skateboarder sitting on the bowl floor beside his board, long shadow, concrete slab seams"),
            dict(id="C", seed=1803, prompt="a skateboarder in the air above the rim of a concrete bowl, deep blue sky, sunlit, low angle")],
)

if __name__ == "__main__":
    L.main(DESIGN)
