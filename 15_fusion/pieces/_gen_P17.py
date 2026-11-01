"""P17 Stairwell (v005): 3-shot chain with hard cuts on the downbeats, 3D type flown through, high-key white + graphite + one cobalt. 120 bpm.
Beats follow 15_fusion/v005/DESIGNS.md "P17 Stairwell" (CLIMB 1.0 / THE SPIRAL 4.0 / FLOOR BY FLOOR 7.0 / INTO WHITE 10.0), shots + grid from MEDIA.md.
Lego: fly_through_3d_type (push_through, then follow, dives through the P counter), contour_hand_off (braid via the handrail matte, fan along the floor
ISO lines), half_front_half_behind (handrail in front), window_ending (INTO WHITE tunnels), chain_through_cards (multiplane, one move per shot).
    python 15_fusion/pieces/_gen_P17.py [--check]"""
import _v005_lib as L
from _v005_lib import port

HD = ("hold_drift", dict(fill=0.58, comp=[0.0, 0.0]))      # v005 round 2: the whole word inside the frame at the hold (fill 0.7 + the off-centre comp clipped the B / S / R)

DESIGN = dict(
    id="P17", slug="stairwell", bpm_design=120.0, anchor=(0.0, 0.273), pace="mid", palette="high_key_one_accent", type_role="solid_3d",
    camera_lead="push_through", technique="built_in", mood="warm", seed=17, cards=4, camera_extra={"subject_z": 1000.0},
    post=[L.legible_camera], extra_chain=[dict(module="follow", window="trans:1"), dict(module="push_through", word=3)],
    blocks=["fly_through_3d_type", "contour_hand_off", "half_front_half_behind", "window_ending", "chain_through_cards"],
    scene="a white concrete spiral staircase, a long white corridor with slot windows, a bright rooftop landing",
    why="v005 design P17: 3 shots, 3D type the camera flies through (push_through then follow), cuts on downbeats (DESIGNS.md)",
    deferred=["push_through: ONE dive, through the counter of the LAST word (INTO WHITE, [release - 0.5 s, end of the piece], to_next off): every word sits on the same plane, so a dive "
              "through CLIMB / THE SPIRAL ended behind the plane and U-turned through the next word (v005 g03: THE SPIRAL mirrored, then seen from inside a letter). "
              "The holds of the other words are hold_drift (default tiling), follow runs in the transition into FLOOR BY FLOOR",
              "hard cut 6.0 mid-twist (the cable leaves the handrail of s1 and lands on the light-bar edge of s2): the twist carries the contour ports through the cut, "
              "but the cut lands on the downbeat of the shot spans, not inside the transition (block cut_mid_transition is spec only)",
              "CLIMB construction triangle (M3b) and THE SPIRAL letters in cells (P9): full layouts stand in (graphite type on a flat shape needs a dark-on-light tint, the recipe tints bright)",
              "FLOOR BY FLOOR window shows the stairwell of s1 (second plate knockout): the window matte treatment dims / brightens the SAME plate"],
    words=[
        dict(text="CLIMB", font="Anybody", module="staircase", hit=1.0, hold=2.0, exit_s=0.8, gap=0.5, io=("lines", "lines"), layout="full",
             entry=port("edge", "up"), matte="front", via=port("matte", "up"), transition="braid", slot=1, camera=HD),
        dict(text="THE SPIRAL", lines=["THE", "SPIRAL"], font="Nixie One", module="hinge_unfold", hit=4.0, hold=1.5, exit_s=0.5, gap=0.5,
             io=("lines", "lines"), layout="full", matte="front", via=port("edge", "up"), transition="twist", slot=1, camera=HD),
        dict(text="FLOOR BY FLOOR", lines=["FLOOR", "BY", "FLOOR"], font="Major Mono Display", module="xray_unfold", hit=7.0, hold=2.0, exit_s=0.8, gap=0.5,
             io=("lines", "lines"), layout="full", matte="window", via=port("iso", "along"), transition="fan", slot=1, camera=HD,
             params=dict(depth=0.8, slices=9)),
        dict(text="INTO WHITE", lines=["INTO", "WHITE"], font="Blackout", module="fly_into", hit=10.0, hold=3.5, exit_s=0.5, io=("lines", "wipe"),
             layout="full", matte="glow", exit=port("edge", "along"), slot=1, camera=HD, params=dict(tilt=0.4, rings=12)),
    ],
    shots=[dict(id="s1", still="A", frames=153, span=151, seed=171,
                prompt="the camera slowly rises straight up the white spiral staircase in one continuous move toward the round skylight, steady, smooth, "
                       "soft daylight, minimal, cinematic"),
           dict(id="s2", still="B", frames=97, span=95, seed=172,
                prompt="the camera tracks smoothly forward along the white corridor in one continuous move, bars of sunlight slide past on the floor"),
           dict(id="s3", still="C", frames=97, span=90, seed=173,
                prompt="static locked-off camera on the bright rooftop landing, a thin curtain of haze drifts, pale sky")],
    stills=[dict(id="A", seed=1701, prompt="looking straight up a white concrete spiral staircase toward a round skylight, bright soft daylight, clean curved "
                                           "handrail, high-key, minimal architecture, vertical frame"),
            dict(id="B", seed=1702, prompt="long white concrete corridor with tall vertical slot windows throwing bars of sunlight across the floor, high-key, vertical frame"),
            dict(id="C", seed=1703, prompt="top landing of a white stairwell opening onto an empty bright white rooftop and pale sky, minimal, vertical frame")],
)

if __name__ == "__main__":
    L.main(DESIGN)
