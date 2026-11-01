"""P16 Below (v005): ONE locked-off 16 s shot, massive frame-filling type, near-black + one pale-cyan light. 75 bpm design (song g03, tracked 74.84; g01 / g02 were 73.37 and went silent from 14.0 s) -> P16.json.
Beats follow 15_fusion/v005/DESIGNS.md "P16 Below" word by word (landing 0.8 then every downbeat), shot / grid from 15_fusion/v005/MEDIA.md.
Lego: window_ending (DOWN as a window onto the lit water), matte_edge_reveal_wipe (EVERYTHING), half_front_half_behind (THE NOISE / SLOWS),
contour_hand_off (braid VIA the bubble column, spiral_around it, ripple on the ISO lines), chain_through_cards (multiplane, one ease_reversal run).
    python 15_fusion/pieces/_gen_P16.py [--check]"""
import _v005_lib as L
from _v005_lib import port, HOLD_CAM

DESIGN = dict(
    id="P16", slug="below", bpm_design=75.0, anchor=(0.0, 0.0), pace="slow", palette="near_black_one_light", type_role="massive",
    camera_lead="ease_reversal", technique="behind_sheer", mood="calm", seed=16, cards=4,
    post=[L.legible_camera],
    blocks=["window_ending", "matte_edge_reveal_wipe", "contour_hand_off", "chain_through_cards", "half_front_half_behind"],
    scene="deep dark water at night, one narrow shaft of pale light falling from far above, a column of silver bubbles rising through the beam",
    why="v005 design P16: one locked-off shot, massive frame-filling layouts, contour hand-offs, type as the window ending (DESIGNS.md)",
    deferred=["v005 round 2 (song): the g01 / g02 song (seed 1) was silent from 14.0 s while DOWN holds to 15.25 s; song g03 = ACE XL base t2m seed 3 (same prompt, tame_highs deep, exact 16.000 s, end gate passed, music to the last second, "
              "74.84 bpm, downbeats 0 / 3.19 / 6.39 / 9.6 / 12.81); the old one is kept as audio/P16_g02_silent_tail.wav (+ _raw, _grid)",
              "v005 round 2 (legibility): EVERYTHING spine -> full (EVERY / THING; the spine box gave 14 px letters, the rule is >= 7 % of the frame = 45 px), DOWN massive_crop -> full "
              "(D and N were cut), every full hold = hold_drift fill 0.58 without the off-centre comp, yaw 6; SLOWS candy_orbit in Fascinate (the hairline face drowned in the rod copies), 24 copies",
              "rise trail on the bubble column 3.2-6.4 (masktrack rise): not in the piece, it would regenerate s1 with LTX v2v and switch flow_wake off there",
              "ONE type move: ease_reversal runs on EVERYTHING (6.4-8.4 rest) and the other holds use hold_drift in the layout framing (one lateral run across all five words is not planned by the chain)",
              "BELOW 'cropped by the top edge' / SLOWS 'ring of letters': hung_from_top and full layouts stand in (no ring-of-letters layout)"],
    words=[
        dict(text="BELOW", font="Big Shoulders Display", module="slats", hit=0.8, hold=1.4, exit_s=0.8, gap=0.5, io=("wipe", "particles"),
             layout="hung_from_top", entry=port("edge", "up"), exit=port("edge", "up"), via=port("matte", "up"), transition="braid", slot=0),
        dict(text="THE NOISE", lines=["THE", "NOISE"], font="Bungee Hairline", module="depth_echo", hit=3.2, hold=2.0, exit_s=0.8, gap=0.5,
             io=("blobs", "lines"), layout="full", matte="front", via=port("iso", "along"), transition="ripple", slot=1, camera=("hold_drift", dict(fill=0.5, comp=[0.0, 0.0])), yaw=6),
        dict(text="EVERYTHING", lines=["EVERY", "THING"], font="Ultra", module="bold_thin_echo", hit=6.4, hold=2.0, exit_s=0.8, gap=0.5, io=("lines", "lines"),
             layout="full", entry=port("matte", "up"), matte="wipe", via=port("matte", "around"), transition="spiral_around", slot=1,
             camera="ease_reversal", yaw=6),
        dict(text="SLOWS", font="Fascinate", module="candy_orbit", hit=9.6, hold=2.4, exit_s=0.7, gap=0.5, io=("lines", "lines"),
             layout="full", matte="glow", exit=port("edge", "up"), transition="twist", slot=1, camera=HOLD_CAM, yaw=6, params=dict(copies=24, band=2)),
        dict(text="DOWN", font="Blackout", module="depth_tunnel", hit=12.8, hold=2.4, exit_s=0.7, io=("lines", "flicker"),
             layout="full", matte="window", slot=2, camera=HOLD_CAM, yaw=6),
    ],
    shots=[dict(id="s1", still="A", frames=385, span=384, seed=161,
                prompt="static locked-off camera, deep dark water, a narrow shaft of pale light from the surface far above sways very slowly, a column of "
                       "bubbles rises steadily through it, fine silt drifts sideways across the frame, calm, near black, cinematic")],
    stills=[dict(id="A", seed=1601, neg=", people, face, fish",
                 prompt="deep dark water at night, a single narrow shaft of pale light falls from far above, a slow column of silver bubbles rises "
                        "through the beam, fine silt drifting in the light, surroundings almost black, vertical frame, high contrast, cinematic")],
)
if __name__ == "__main__":
    L.main(DESIGN)
