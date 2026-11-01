"""blocks.py - LEGO building blocks: the reusable PASSAGE ideas of the v005 designs (15_fusion/v005/DESIGNS.md, P16-P20) as small declarative
specs, to be combined freely with word modules, chains, camera moves, hold layouts and contour keys. NOT whole-piece presets: a block says only
what it needs, which slots it fills (some lines / a transition / a camera / compose ops / the shot structure) and its parameter ranges.

    BLOCKS[name]                        the spec (see SCHEMA below)
    eligible(name, ctx)                 -> (ok, [reasons])  against a piece context {shots, technique, n_lines, format, sources, subject, min_hold_s}
    choose_blocks(ctx, seed, ...)       -> up to 2 names (the first = dominant) for one piece; honours a Qwen brief's blocks + direction
    batch_blocks(ctxs, seed, briefs)    -> one list per piece, NO two pieces in a batch with the same dominant block (while there are enough)
    line_constraints(names, n)          -> per line {layout, module, camera, io, roles}: allowed sets (None = anything) for assign_batch
    resolve_params(name, seed, tag)     -> {param: value inside its range}
    describe(names, seed, n)            -> what plan_rules stores in piece["plan"]["blocks"] (params, constrained lines, compose templates, status)

status: live = everything it needs exists (the slots are really applied) | partial = the slots below are applied, the listed ``missing``
parts are templates only (recorded in plan.blocks[].compose for the runner) | spec = the registry entry exists, the render side does not
(``missing`` names what is to be built). The planner never claims more than the status says.

SCHEMA (all keys optional except doc / role / status):
  doc, from [pieces], role (the type role it gives: massive | solid_3d | flat_strips | from_footage | small_precise | window | any),
  paces [slow | mid | driving | building | flowing], status, missing [..],
  needs {shots: [single|cut|chain], techniques, sources [matte|depth|edges|flow|contours|cards], subject, min_hold_s, min_lines, max_lines,
         formats, min_shots, opt_in (never chosen on its own: the brief / request names it), prefer [notes: inputs it takes in the better form when they exist]},
  slots {lines {which: all|first|last|middle|even|odd, layout, module, camera, io, roles}, transitions [..], camera [..],
         pan {boundaries: alt, dirs: [down, up, right, left]} (batch_plan picks the boundaries and the directions),
         image_transition {module, between, node, video} (an image-domain transition between consecutive shots, not a type-engine transition),
         compose [{op, when, args}], shots {mode, count}, contour {ports, use}},
  params {name: {lo, hi, default, kind: int|float|choice, choices}}
"""
from __future__ import annotations

import zlib

ROLES = ("massive", "solid_3d", "flat_strips", "from_footage", "small_precise", "window", "any")
PACES = ("slow", "mid", "driving", "building", "flowing")
STATUS = ("live", "partial", "spec")
WHICH = ("all", "first", "last", "middle", "middle_alt", "even", "odd")
GALLERY_CHAINS = (                       # the 13 approved chains of the Module Gallery (artifact-016) + the techy engine chain used in v005
    "chain_flow_bends_demo", "chain_flow_bends_follow", "chain_signal_bends_flow_demo", "chain_signal_bends_flow_follow",
    "chain_signal_circuit_bends_follow", "chain_bends_spiral_flow_follow", "chain_signal_braid_bends_follow", "chain_bends_fan_flow_follow",
    "engine_chain_1_calm", "engine_chain_2_techy", "engine_chain_3_hype", "io_chain_flicker_blobs_twist", "io_chain_wipe_particles_braid")
WINDOW_MODULES = ("depth_tunnel", "fly_into")
LAYOUT_CAMS = ("hold_drift", "follow", "pullback_reveal", "orbit_even", "spin_read_spin", "ease_reversal")


def _p(lo, hi, default, kind="float", **kw):
    return dict(lo=lo, hi=hi, default=default, kind=kind, **kw)


BLOCKS: dict[str, dict] = {
    "strips_persist_across_cuts": dict(
        doc="flat colour strips / blocks stay on screen across the hard cuts and re-snap their angle to the new shot's edges on the downbeat "
            "(the TYPE layer is the continuity over cutting video)",
        **{"from": ["P18"]}, role="flat_strips", paces=["driving", "mid"], status="live",
        needs=dict(shots=["cut"], sources=["edges"], min_shots=3, min_lines=3, min_hold_s=1.4),
        slots=dict(lines=dict(which="even", layout=["solid_strips", "split_sides", "edge_L", "cells"], camera=["hold_drift"]),
                   compose=[dict(op="layout_shapes", when="always", args=dict(persist=True, pose="cut_targets", ease_s=0.25)),
                            dict(op="rigid_pose", when="always", args=dict(layers=["wm", "wl"], follow="strips"))]),
        params=dict(angle_deg=_p(0.0, 10.0, 6.0, unit="max strip tilt deg"), persist_s=_p(0.4, 2.0, 1.0, unit="s past the line's end"),
                    persist_cuts=_p(0, 2, 1, "int", unit="hard cuts the strip survives after its line has gone")),
    ),
    "chain_through_cards": dict(
        doc="a named gallery chain's CameraChain move travels through multiplane depth cards (the plate split by depth into K0 far .. K3 near "
            "in the same 3D camera as the type) with lines running between the cards; the line sits between two cards so nearer cards pass in front",
        **{"from": ["P16", "P17", "P18", "P19", "P20"]}, role="any", paces=["slow", "mid", "flowing", "building"], status="partial",
        missing=["compose op that renders the cards (mograph/render/multiplane.MultiPlane.render) in the recipe; the camera chain through the cards "
                 "(camera dict + word slot) is wired in spec.build_scene"],
        needs=dict(sources=["depth"], min_hold_s=1.4),
        slots=dict(lines=dict(which="all", camera=list(LAYOUT_CAMS)),
                   camera_plate=dict(plate="multiplane", cards=4), slot=dict(which="all", between="K1|K2"),
                   compose=[dict(op="multiplane_cards", when="always", args=dict(cards=4, between="K1|K2"))]),
        params=dict(cards=_p(3, 5, 4, "int"), chain=_p(None, None, "engine_chain_1_calm", "choice", choices=list(GALLERY_CHAINS))),
    ),
    "type_from_sparks": dict(
        doc="type built from the footage's OWN particles / edges: the plate's sparks (edge blobs, tracked) become the streaks that land on glyph "
            "sites, letters made of the footage's edges",
        **{"from": ["P19", "P08"]}, role="from_footage", paces=["building", "driving"], status="partial",
        missing=["edge blobs of the plate as particle sources for the `particles` module (ops exist: trace_edges, type_from_edges)"],
        needs=dict(sources=["edges", "flow"], min_hold_s=1.4),
        slots=dict(lines=dict(which="middle_alt", module=["tangent_spray", "voxel_orbit", "spectrum"], io=["flicker", "blobs", "lines"]),
                   compose=[dict(op="type_from_edges", when="form", args=dict(source="trace_edges"))]),
        params=dict(streaks=_p(800, 2400, 2000, "int"), edge_low=_p(0.02, 0.06, 0.03)),
    ),
    "window_ending": dict(
        doc="the last word is a window into the plate: the plate shows only inside the letters, outside goes a flat ground (P10 ground flip), "
            "held long, exit flicker to black",
        **{"from": ["P16", "P17", "P10"]}, role="window", paces=["slow", "mid", "flowing"], status="partial",
        missing=["auto-insert of knockout (+ flat ground) into the recipe chain: the template below is for the runner"],
        needs=dict(min_hold_s=2.0, min_lines=2),
        slots=dict(lines=dict(which="last", layout=["counters_window", "massive_crop", "full"], io=["flicker", "lines", "blobs", "wipe"]),
                   matte=dict(which="last", mode="window"), compose=[dict(op="knockout", when="hold", args=dict(plate="flat", ground="@bg"))]),
        params=dict(hold_s=_p(2.0, 3.6, 2.4, unit="s"), ground=_p(None, None, "flat_bg", "choice", choices=["flat_bg", "second_plate", "flat_accent"])),
    ),
    "window_other": dict(
        doc="P10-style window onto ANOTHER shot: while a word holds, its letters show the next shot (or a named one) and the word's own shot stays "
            "outside, the subject of the plate stepping out over both (compose op window_cut mode other; design key words[].window_source). Heavy "
            "liked face, cap height >= 30 % of the frame, the two shots must differ in luma or colour",
        **{"from": ["P10"]}, role="window", paces=["slow", "mid", "driving", "building", "flowing"], status="partial",
        missing=["no GPU render of the two-shot compose yet (wired: MographCompose and box_stages.cmd_compose pass the neighbour shot as `others`; "
                 "the design step writes compose.cuts / words[].window_source through design_options.validate + to_piece)"],
        needs=dict(shots=["cut", "chain"], min_shots=2, min_lines=1, min_hold_s=1.6, opt_in=True),
        slots=dict(lines=dict(which="middle_alt", layout=["massive_crop", "counters_window", "full"]),
                   compose=[dict(op="window_cut", when="hold", args=dict(mode="other", b="$next", subject="outside", min_contrast=0.22))]),
        params=dict(exit_s=_p(0.3, 1.0, 0.5, unit="s the window fades when the hold ends"), words=_p(1, 2, 1, "int", unit="window_source words per piece")),
    ),
    "window_cut": dict(
        doc="text-window reveal cut: over shot A the word is a window onto shot B, then the letters GROW (distance-field dilation) until B fills the "
            "frame; that is the cut, no crossfade at the boundary (compose op window_cut mode cut; design key compose.cuts[] cut=window_cut)",
        **{"from": ["P10", "user 2026-10-28"]}, role="window", paces=["slow", "mid", "driving", "building"], status="partial",
        missing=["no GPU render of the two-shot compose yet (wired: MographCompose and box_stages.cmd_compose pass the neighbour shot as `others`; "
                 "the design step writes compose.cuts / words[].window_source through design_options.validate + to_piece)"],
        needs=dict(shots=["cut", "chain"], min_shots=2, min_lines=1, min_hold_s=1.6, opt_in=True),
        slots=dict(lines=dict(which="middle_alt", layout=["massive_crop", "counters_window", "full"]),
                   cut=dict(kind="window_cut", boundaries=1, owns_boundary=True),
                   compose=[dict(op="window_cut", when="hold", args=dict(mode="cut", exit="grow", min_contrast=0.22))]),
        params=dict(grow_s=_p(0.4, 1.4, 0.7, unit="s the window takes to fill the frame"), count=_p(1, 2, 1, "int", unit="window cuts per piece")),
    ),
    "window_zoom": dict(
        doc="zoom through a letter (2026-10-30): the holding word is a window onto shot B over shot A; the camera then SCALES INTO ONE LETTER (letters, ground and A's subject matte scale together "
            "about the letter's thickest point, which glides to the centre; B stays still inside) until the letter covers the frame and B fills it; that is the cut, no crossfade "
            "(compose op window_cut mode zoom; design key compose.cuts[] cut=window_zoom, zoom_s >= 0.5 s)",
        **{"from": ["user 2026-10-30"]}, role="window", paces=["slow", "mid", "driving", "building"], status="partial",
        missing=["no GPU box render of the two-shot compose yet (the geometry and a CPU draft are tested; the type matte must come from the box type pass)"],
        needs=dict(shots=["cut", "chain"], min_shots=2, min_lines=1, min_hold_s=1.6, opt_in=True),
        slots=dict(lines=dict(which="middle_alt", layout=["massive_crop", "counters_window", "full"]),
                   cut=dict(kind="window_zoom", boundaries=1, owns_boundary=True),
                   compose=[dict(op="window_cut", when="hold", args=dict(mode="zoom", exit="zoom", min_contrast=0.22))]),
        params=dict(zoom_s=_p(0.5, 1.6, 0.9, unit="s the camera takes to dive into the letter (>= 0.5 s so it can be seen)"), count=_p(1, 2, 1, "int", unit="window cuts per piece")),
    ),
    "window_swap": dict(
        doc="text-window swap cut: B inside the letters with A outside; on the cut beat (downbeat / chord change / dropout) they SWAP (B the whole "
            "picture, A inside the letters), then the letters leave (fade / shrink / io) and B remains; optional freeze on the swap; no crossfade "
            "(compose op window_cut mode swap; design key compose.cuts[] cut=window_swap)",
        **{"from": ["P10", "user 2026-10-28"]}, role="window", paces=["slow", "mid", "driving", "building"], status="partial",
        missing=["no GPU render of the two-shot compose yet (wired: MographCompose and box_stages.cmd_compose pass the neighbour shot as `others`; "
                 "the design step writes compose.cuts / words[].window_source through design_options.validate + to_piece)"],
        needs=dict(shots=["cut", "chain"], min_shots=2, min_lines=1, min_hold_s=1.6, opt_in=True),
        slots=dict(lines=dict(which="middle_alt", layout=["massive_crop", "counters_window", "full"]),
                   cut=dict(kind="window_swap", boundaries=1, owns_boundary=True),
                   compose=[dict(op="window_cut", when="hold", args=dict(mode="swap", exit="fade", min_contrast=0.22))]),
        params=dict(exit_s=_p(0.3, 1.0, 0.5, unit="s the letters take to leave"), freeze_s=_p(0.0, 0.5, 0.0, unit="s freeze on the swap (0 = none)"),
                    count=_p(1, 2, 1, "int", unit="window cuts per piece")),
    ),
    "fill_silhouette_grow": dict(
        doc="a word fills the subject's silhouette (solid inside the MATTE, hollow outline outside) and grows with it as it moves",
        **{"from": ["P18", "P19"]}, role="from_footage", paces=["driving", "building"], status="partial",
        missing=["glyphs re-fitted to the live matte contour every frame (the `fill` matte treatment shows the line inside the silhouette only; "
                 "it reads faint on thin type: heavy modules)"],
        needs=dict(techniques=["behind", "behind_sheer"], sources=["matte"], subject=True, min_hold_s=1.4),
        slots=dict(lines=dict(which="last", module=["scale_clash", "tangent_spray", "lanes", "solid_hollow"]),
                   contour=dict(ports=["matte_outline"], use="entry"), matte=dict(which="last", mode="fill"),
                   compose=[dict(op="silhouette_fill", when="hold", args=dict(grow=True))]),
        params=dict(grow=_p(0.05, 0.3, 0.15), outline_px=_p(2.0, 6.0, 4.0)),
    ),
    "edge_run_small_type": dict(
        doc="precise small type running along the footage's edges / depth lines: one line, cap height >= 4 % of the frame height "
            "(26 px at 640), woven into the contours",
        **{"from": ["P20"]}, role="small_precise", paces=["flowing", "slow", "mid"], status="live",
        needs=dict(sources=["edges", "contours"], min_hold_s=1.4, min_lines=3),
        slots=dict(lines=dict(which="middle_alt", layout=["small_line"], module=["circuit", "pipework", "lanes", "topo_tower", "spectrum"]),
                   contour=dict(ports=["matte_outline", "depth_iso"], use="entry"),
                   compose=[dict(op="edge_run_type", when="hold", args=dict(source="luma_edges", cap_frac=0.05, min_px_at_640=26, track=0.1))]),
        params=dict(cap_frac=_p(0.04, 0.07, 0.05), min_px_at_640=_p(26, 40, 26, "int")),
    ),
    "fly_through_3d_type": dict(
        doc="3D type objects the camera flies through: the move lines up on a letter's counter, dives through it toward the next scene, "
            "the cut / transition lands on the far side (counter-window modules, extruded type)",
        **{"from": ["P17"]}, role="solid_3d", paces=["mid", "building"], status="live",
        needs=dict(min_hold_s=1.6, min_lines=2),
        slots=dict(lines=dict(which="middle_alt", module=["fly_into", "depth_tunnel", "staircase", "hinge_unfold", "xray_unfold", "hollow_tubes"],
                              camera=["push_through", "follow"], layout=["counters_window", "full", "massive_crop"]),
                   transitions=["twist", "braid", "fan"]),
        params=dict(counter=_p(0, 2, 0, "int"), dive_s=_p(0.8, 1.8, 1.2, unit="s")),
    ),
    "half_front_half_behind": dict(
        doc="part of the line sits in front of the subject, part behind (sandwich with front=; legible keeps the read)",
        **{"from": ["P16", "P17", "P18", "P19"]}, role="any", paces=["slow", "mid", "driving", "building"], status="live",
        needs=dict(techniques=["behind", "behind_sheer"], sources=["matte"], subject=True),
        slots=dict(lines=dict(which="all"), contour=dict(ports=["matte_outline"], use="via"), matte=dict(which="all", mode="front"),
                   compose=[dict(op="sandwich", when="hold", args=dict(front="$type_matte", matte_threshold=0.5))]),
        params=dict(front_share=_p(0.3, 0.7, 0.5)),
    ),
    "matte_edge_reveal_wipe": dict(
        doc="letters appear only where the subject's matte edge has swept: an edge-following reveal wipe on the entry",
        **{"from": ["P16", "P18", "P20"]}, role="any", paces=["slow", "driving", "flowing"], status="live",
        needs=dict(sources=["matte", "edges"], subject=True),
        slots=dict(lines=dict(which="even", io=["wipe"]), contour=dict(ports=["matte_outline"], use="entry"), matte=dict(which="even", mode="wipe"),
                   compose=[dict(op="edge_reveal", when="form", args=dict(sweep_s=1.0))]),
        params=dict(sweep_s=_p(0.6, 1.6, 1.0, unit="s")),
    ),
    "contour_hand_off": dict(
        doc="the transition rides a footage contour: A exits INTO a contour, the strands travel VIA it (spiral_around the subject, braid along "
            "channels) and B enters FROM the next one",
        **{"from": ["P16", "P17", "P18", "P19", "P20"]}, role="any", paces=["slow", "mid", "driving", "building", "flowing"], status="partial",
        missing=["contour bundles as transition endpoints (tools/fusion/contour_ports.py, being built)"],
        needs=dict(sources=["contours"], min_lines=2),
        slots=dict(lines=dict(which="all"), transitions=["spiral_around", "braid", "circuit_board", "fan", "ripple"],
                   contour=dict(ports=["matte_outline", "depth_iso"], use="via")),
        params=dict(object=_p(None, None, "matte", "choice", choices=["matte", "depth_iso", "edge"])),
    ),
    "ground_flip": dict(
        doc="the flat ground flips between lines: type as window over a flat colour on one line, sandwich over the plate on the next",
        **{"from": ["P10", "P16"]}, role="window", paces=["slow", "mid", "driving"], status="live",
        needs=dict(min_lines=3),
        slots=dict(lines=dict(which="odd", layout=["massive_crop", "counters_window", "full"]),
                   compose=[dict(op="knockout", when="hold", args=dict(plate="flat", ground="@bg"))]),
        params=dict(ground_alt=_p(None, None, "flat_bg", "choice", choices=["flat_bg", "flat_accent"])),
    ),
    "beat_glow": dict(
        doc="outline glow pulses on the beats (add_glow with the onset curve) on the lines' matte",
        **{"from": ["P16", "P17", "P19", "P20"]}, role="any", paces=["slow", "mid", "driving", "building", "flowing"], status="live",
        needs=dict(),
        slots=dict(matte=dict(which="all", mode="glow"), compose=[dict(op="add_glow", when="hold", args=dict(mask="$type_matte", intensity=dict(curve="onset", lo=0.1, hi=0.6)))]),
        params=dict(radius=_p(14, 30, 22, "int"), hi=_p(0.4, 0.8, 0.6)),
    ),
    "rise_trail": dict(
        doc="a masked motion track makes the subject's element rise (never downward on people): the bubble column / smoke climbs while the "
            "line forms, flow_wake off for that window",
        **{"from": ["P16"]}, role="any", paces=["slow", "flowing"], status="live",
        needs=dict(sources=["matte"], subject=True, min_hold_s=1.4),
        slots=dict(lines=dict(which="middle"), compose=[dict(op="mask_track", when="form", args=dict(style="rise"))]),
        params=dict(rise_s=_p(2.0, 4.0, 3.2, unit="s")),
    ),
    "locked_plate_move": dict(
        doc="the plate is locked off and the TYPE camera does the one move (ease_reversal / tilt_to_plan / pullback_reveal): card parallax and "
            "type are the only motion, nothing fights",
        **{"from": ["P16", "P20"]}, role="any", paces=["slow", "flowing"], status="live",
        needs=dict(shots=["single", "cut"]),
        slots=dict(camera=["ease_reversal", "tilt_to_plan", "pullback_reveal"], lines=dict(which="all", camera=["ease_reversal", "tilt_to_plan", "pullback_reveal", "hold_drift"]),
                   shots=dict(mode="single", prompt="static locked-off camera")),
        params=dict(move=_p(None, None, "ease_reversal", "choice", choices=["ease_reversal", "tilt_to_plan", "pullback_reveal"])),
    ),
    "orbit_ring_around_subject": dict(
        doc="a ring of letters / cubes orbits the subject (circle of type, M5a): back half behind the subject, front half in front",
        **{"from": ["P16", "P19"]}, role="solid_3d", paces=["slow", "building"], status="live",
        needs=dict(sources=["matte"], subject=True, min_hold_s=1.6),
        slots=dict(lines=dict(which="middle_alt", module=["candy_orbit", "voxel_orbit"], camera=["orbit_even", "spin_read_spin"], layout=["full"])),
        params=dict(sweep_deg=_p(60, 120, 100, unit="deg")),
    ),
    "colour_shift_on_hold": dict(
        doc="the colour of a held line shifts through the hold (amber cools to steel blue): a graded tint timeline on the beat",
        **{"from": ["P19"]}, role="any", paces=["building", "mid"], status="partial",
        missing=["per-line tint timeline op (grade / tint_type keyframes exist; not wired per hold)"],
        needs=dict(min_hold_s=1.8),
        slots=dict(lines=dict(which="odd"), compose=[dict(op="tint_type", when="hold", args=dict(timeline="warm_to_cool"))]),
        params=dict(shift=_p(0.3, 1.0, 0.7)),
    ),
    "gallery_chain_open": dict(
        doc="a named gallery chain drives the opening words exactly (its io styles, transitions and follow camera), the rest of the piece "
            "continues with free choices",
        **{"from": ["P16", "P17", "P18", "P19", "P20"]}, role="any", paces=["slow", "mid", "driving", "building", "flowing"], status="partial",
        missing=["replay of a stored gallery chain by name (14_modules/gallery/build_gallery.py); the planner takes its transitions / io only"],
        needs=dict(min_lines=3),
        slots=dict(lines=dict(which="first"), transitions=["twist", "braid", "fan", "ripple", "spiral_around", "circuit_board"]),
        params=dict(chain=_p(None, None, "io_chain_wipe_particles_braid", "choice", choices=list(GALLERY_CHAINS)), words=_p(2, 4, 3, "int")),
    ),
    "cut_mid_transition": dict(
        doc="a hard cut lands in the middle of a transition: the strands leave the old shot's contour and land on the new shot's edge on the downbeat",
        **{"from": ["P17", "P18"]}, role="any", paces=["driving", "mid"], status="spec",
        missing=["transition endpoints on a per-shot contour across a hard cut (contour_ports + a cut-aware Sequence gap)"],
        needs=dict(shots=["cut"], sources=["contours"], min_shots=2, min_lines=2),
        slots=dict(transitions=["twist", "fan", "circuit_board", "braid"], contour=dict(ports=["depth_iso", "matte_outline"], use="via")),
        params=dict(cut_at=_p(0.3, 0.7, 0.5, unit="share of the gap")),
    ),
    "camera_pan_transition": dict(
        doc="the camera travel IS the transition: word A holds, the camera travels down / up / right / left, A leaves the frame on the opposite "
            "side as a world-locked object would and B (placed one frame-span further along) arrives as the camera settles, B's hit on a downbeat; "
            "the footage plate is panned by the same on-screen displacement with directional motion blur, and the shots' LTX prompts name the "
            "matching camera move (tilt down / pan left ...)",
        **{"from": ["v005 review 2026-10-27"]}, role="any", paces=["slow", "mid", "driving", "building", "flowing"], status="live",
        needs=dict(min_lines=2, min_hold_s=1.4, opt_in=True),
        slots=dict(lines=dict(which="all"), pan=dict(boundaries="alt", dirs=["down", "up", "right", "left"]),
                   compose=[dict(op="plate_pan", when="travel", args=dict(source="type_camera", mode="wrap", shutter=0.5))]),
        params=dict(beats=_p(2, 4, 2, "int", unit="beats of travel"), count=_p(1, 3, 2, "int", unit="pan boundaries per piece")),
    ),
    "depth_pivot_chunks": dict(
        doc="the depth pivot-chunk IMAGE transition between consecutive shots (17_depth-dolly v013, the approved look): both shots become depth scenes "
            "of chunky camera-facing cards (the subject one cut-out card), one camera pivots round A's volume, B swings in over the top turned 45 deg "
            "and lands flat; LIVE VIDEO is the default (the shot's motion clip + per-frame Alpha-Gen mattes play on the cards, a single still only "
            "when no clip exists); seeded, density = `chunks`",
        **{"from": ["17_depth-dolly v013"]}, role="any", paces=["slow", "mid", "driving", "building", "flowing"], status="partial",
        missing=["splice side is built (tools/fusion/pivot_cut.py: compose.cuts[] cut=depth_pivot, hard-cut ownership, node MographPivotCut) but the node is NOT in "
                 "any build_graphs graph yet (insert between the flow node and MographCompose), no planner / design step writes the cut, the shots' mattes do not "
                 "follow the replaced frames (matte pass depth_orbit --matte-pass not wired), no GPU-box run; type planes (v014 `type`) in the scene are API-only"],
        needs=dict(shots=["cut", "chain"], sources=["matte"], min_shots=2, opt_in=True,
                   prefer=["video: the shot's motion clip, not a still", "mattes: per-frame Alpha-Gen matte sequence, not a single matte"]),
        slots=dict(image_transition=dict(module="depth_pivot", between="consecutive_shots", node="MOTNGRFX_DepthPivotTransition",
                                         video="auto"),
                   cut=dict(kind="depth_pivot", boundaries=1, owns_boundary=True)),
        params=dict(chunks=_p(10, 400, 32, "int", unit="background cards per image (+ the subject as one card); card size scales with it"),
                    in_angle=_p(10.0, 80.0, 45.0, unit="deg B is turned when it enters"), out_angle=_p(5.0, 80.0, 40.0, unit="deg A has turned when B enters"),
                    travel=_p(4, 120, 36, "int", unit="frames of the eased move"), depth_ratio=_p(2.0, 12.0, 6.0),
                    size_lo=_p(0.5, 8.0, 1.5, unit="advanced: smallest card = ref x this"), size_hi=_p(0.5, 8.0, 4.0, unit="advanced: largest card = ref x this")),
    ),
}
NAMES = tuple(BLOCKS)
PAN_BLOCK = "camera_pan_transition"


# ================================================================================================== helpers
def _rank(seed, tag, *parts) -> int:
    return zlib.crc32(f"{int(seed)}:{tag}:{':'.join(str(p) for p in parts)}".encode("utf-8"))


def validate_block(name: str) -> list[str]:
    """problems of one registry entry ([] = well formed)."""
    b, bad = BLOCKS[name], []
    for k in ("doc", "role", "status"):
        if not b.get(k):
            bad.append(f"{name}: no {k}")
    if b.get("role") not in ROLES:
        bad.append(f"{name}: role {b.get('role')!r}")
    if b.get("status") not in STATUS:
        bad.append(f"{name}: status {b.get('status')!r}")
    if b.get("status") != "live" and not b.get("missing"):
        bad.append(f"{name}: status {b.get('status')} needs a `missing` list")
    for pc in b.get("paces", []):
        if pc not in PACES:
            bad.append(f"{name}: pace {pc!r}")
    ls = (b.get("slots") or {}).get("lines") or {}
    if ls and ls.get("which", "all") not in WHICH:
        bad.append(f"{name}: lines.which {ls.get('which')!r}")
    if ls.get("layout") and ls.get("camera"):
        from tools.fusion import hold_layouts as hl
        shapes = [n for n in ls["layout"] if n in hl.NAMES and hl.build(n, 0, "v", hl.LAYOUTS[n]["fmts"][0])["shapes"]]
        if shapes and "hold_drift" not in ls["camera"]:
            bad.append(f"{name}: layouts with shapes ({shapes}) need hold_drift in the line cameras")
    for k, v in (b.get("params") or {}).items():
        if v["kind"] == "choice":
            if v["default"] not in v.get("choices", []):
                bad.append(f"{name}.{k}: default not in choices")
        elif not (v["lo"] <= v["default"] <= v["hi"]):
            bad.append(f"{name}.{k}: default outside [{v['lo']}, {v['hi']}]")
    return bad


def _imports():
    from tools.fusion import hold_layouts as hl
    from tools.fusion import plan_rules as pr
    return hl, pr


def known_sets() -> dict:
    """the sets the slot constraints must come from (registered modules, transitions, cameras, io styles, layouts)."""
    from tools.fusion import batch_plan as bp
    from tools.fusion import hold_layouts as hl
    from tools.fusion import plan_rules as pr
    from mograph.modules import io as _io
    from mograph.modules.core import list_modules
    bp.io_styles()                                       # imports mograph.modules (registers words / transitions / io / cameras)
    return {"module": set(list_modules("word")), "transitions": set(pr.transition_names()) | {"cut"}, "camera": set(bp.cameras()),       # the REGISTERED sets: the batch draws fewer (batch_plan.word_modules / io_styles drop the banned speckle ones)
            "io": set(_io.list_styles()), "layout": set(hl.NAMES)}


# ================================================================================================== eligibility
def piece_ctx(shots: str = "cut", technique: str = "built_in", n_lines: int = 5, fmt: str = "9:16", subject: bool | None = None,
              min_hold_s: float = 1.6, n_shots: int | None = None, tier: str = "mid", holds: list | None = None) -> dict:
    """the context blocks are matched against. sources = what the footage side can supply to this piece: edges / flow / contours always
    (traced from the footage), a matte with the matte techniques, depth with built_in / behind (depth control), cards only when the multiplane
    plate exists (it does not yet)."""
    src = {"edges", "flow", "contours"}
    if technique in ("behind", "behind_sheer"):
        src |= {"matte", "depth"}
    if technique == "built_in":
        src |= {"depth"}
    if subject is None:
        subject = technique in ("behind", "behind_sheer")
    return {"shots": shots, "technique": technique, "n_lines": int(n_lines), "format": fmt, "sources": sorted(src), "subject": bool(subject),
            "min_hold_s": float(min_hold_s), "n_shots": n_shots if n_shots is not None else (1 if shots == "single" else 4), "tier": tier,
            "holds": [float(h) for h in (holds or [])]}


def eligible(name: str, ctx: dict) -> tuple[bool, list[str]]:
    """does the piece context supply what the block needs? -> (ok, reasons it does not)."""
    n, why = BLOCKS[name].get("needs") or {}, []
    if n.get("opt_in") and name not in (ctx.get("opt_in") or ()):
        why.append("opt-in block: name it in the brief / request")
    if n.get("shots") and ctx["shots"] not in n["shots"]:
        why.append(f"shots {ctx['shots']} not in {n['shots']}")
    if n.get("techniques") and ctx["technique"] not in n["techniques"]:
        why.append(f"technique {ctx['technique']} not in {n['techniques']}")
    miss = [s for s in n.get("sources", []) if s not in ctx["sources"]]
    if miss:
        why.append(f"no source: {', '.join(miss)}")
    if n.get("subject") and not ctx["subject"]:
        why.append("needs a subject")
    if n.get("min_lines") and ctx["n_lines"] < n["min_lines"]:
        why.append(f"{ctx['n_lines']} lines < {n['min_lines']}")
    if n.get("max_lines") and ctx["n_lines"] > n["max_lines"]:
        why.append(f"{ctx['n_lines']} lines > {n['max_lines']}")
    if n.get("min_hold_s"):                           # the hold of the lines the block uses (all lines without a line slot / without per-line holds)
        ls = (BLOCKS[name].get("slots") or {}).get("lines") or {}
        holds = ctx.get("holds") or []
        used = [holds[i] for i in _which_lines(ls.get("which", "all"), len(holds)) if i < len(holds)] if (holds and ls) else []
        mh = min(used) if used else ctx["min_hold_s"]
        if mh + 1e-9 < n["min_hold_s"]:
            why.append(f"hold {mh:.1f}s < {n['min_hold_s']}s")
    if n.get("min_shots") and ctx["n_shots"] < n["min_shots"]:
        why.append(f"{ctx['n_shots']} shots < {n['min_shots']}")
    if n.get("formats") and ctx["format"] not in n["formats"]:
        why.append(f"format {ctx['format']}")
    return (not why), why


def _which_lines(which: str, n: int) -> list[int]:
    return {"all": list(range(n)), "first": [0], "last": [n - 1], "middle": list(range(1, n - 1)) or [0],
            "middle_alt": list(range(1, n - 1, 2)) or [n // 2],
            "even": list(range(0, n, 2)), "odd": list(range(1, n, 2)) or [n - 1]}[which]


def compatible(a: str, b: str, n: int = 6) -> bool:
    """can two blocks share a piece? their line slots must leave every shared line at least one allowed layout / module / camera / io, their
    shot structures must agree."""
    sa, sb = BLOCKS[a].get("slots") or {}, BLOCKS[b].get("slots") or {}
    ma, mb = (sa.get("shots") or {}).get("mode"), (sb.get("shots") or {}).get("mode")
    if ma and mb and ma != mb:
        return False
    for mode, other in ((ma, b), (mb, a)):                                      # a block that fixes the shot mode vs one that needs another
        need = (BLOCKS[other].get("needs") or {}).get("shots")
        if mode and need and mode not in need:
            return False
    ca, cb = line_constraints([a], n), line_constraints([b], n)
    for i in range(n):
        for k in ("layout", "module", "camera", "io"):
            x, y = ca[i].get(k), cb[i].get(k)
            if x is not None and y is not None and not (set(x) & set(y)):
                return False
    return True


def line_constraints(names, n: int) -> list[dict]:
    """per line {layout, module, camera, io, roles}: the intersection of what the blocks allow (None = any)."""
    out = [dict(layout=None, module=None, camera=None, io=None, roles=None) for _ in range(n)]
    for nm in names:
        sl = BLOCKS[nm].get("slots") or {}
        ls = sl.get("lines") or {}
        cam = sl.get("camera")
        for i in _which_lines(ls.get("which", "all"), n) if ls else []:
            out[i]["io_side"] = "out" if ls.get("which") == "last" else "in"      # the io constraint binds the exit of a last line, else the entry
            for k in ("layout", "module", "camera", "io", "roles"):
                v = ls.get(k)
                if v is None:
                    continue
                out[i][k] = list(v) if out[i][k] is None else [x for x in out[i][k] if x in v]
        if cam and not ls.get("camera"):
            for i in range(n):
                out[i]["camera"] = list(cam) if out[i]["camera"] is None else [x for x in out[i]["camera"] if x in cam]
    return out


def matte_for_lines(names, n: int, modes=None) -> dict[int, str]:
    """{line: matte treatment} the blocks ask for (the word spec `"matte"` key; contour_ports.MATTE_MODES). Later blocks do not override earlier
    ones; modes the piece cannot do (not in ``modes``) are dropped."""
    out: dict[int, str] = {}
    for nm in names:
        m = (BLOCKS[nm].get("slots") or {}).get("matte")
        if not m or (modes is not None and m["mode"] not in modes):
            continue
        for i in _which_lines(m.get("which", "all"), n):
            out.setdefault(i, m["mode"])
    return out


def transition_constraints(names, n_boundaries: int) -> list[list[str] | None]:
    """per boundary the transitions the blocks allow (None = any)."""
    out: list = [None] * n_boundaries
    for nm in names:
        tr = (BLOCKS[nm].get("slots") or {}).get("transitions")
        if tr:
            for b in range(n_boundaries):
                out[b] = list(tr) if out[b] is None else [x for x in out[b] if x in tr]
    return out


# ================================================================================================== choosing
def _score(name: str, direction: dict, seed, tag) -> float:
    b = BLOCKS[name]
    s = 0.0
    role = (direction or {}).get("type_role")
    if role and b["role"] == role:
        s += 2.0
    if (direction or {}).get("pace") in b.get("paces", []):
        s += 1.0
    cam = (direction or {}).get("camera")
    cams = set(((b.get("slots") or {}).get("camera") or [])) | set((((b.get("slots") or {}).get("lines") or {}).get("camera") or []))
    if cam and cam in cams:
        s += 1.0
    return s + (_rank(seed, "block", tag, name) % 1000) / 1000.0 * 0.9           # noise only breaks ties between equal scores


def choose_blocks(ctx: dict, seed: int = 0, brief: dict | None = None, taken: set | None = None, tag: str = "", k: int = 2) -> list[str]:
    """up to k blocks for one piece, the first is the dominant. A Qwen brief's blocks go first (if eligible), then the best eligible by the
    brief's direction (type_role / pace / camera) and a seeded tie-break. Blocks already dominant elsewhere in the batch (taken) come last;
    the second block must be compatible with the first and have a different role. Live / partial blocks are preferred over spec ones."""
    taken = set(taken or ())
    brief = brief or {}
    direction = brief.get("direction") or {}
    forced = [n for n in (brief.get("blocks") or []) if n in BLOCKS and eligible(n, dict(ctx, opt_in=(n,)))[0]]       # naming an opt-in block opts in
    pool = [n for n in NAMES if n not in forced and eligible(n, ctx)[0]]
    pool.sort(key=lambda n: (n in taken, BLOCKS[n]["status"] == "spec", -_score(n, direction, seed, tag), n))
    out: list[str] = []
    for n in forced + pool:
        if len(out) >= k:
            break
        if n in out or (out and (BLOCKS[n]["role"] == BLOCKS[out[0]]["role"] != "any" or not compatible(out[0], n, ctx["n_lines"]))):
            continue
        if not out and n in taken and n not in forced:      # prefer a fresh dominant; fall back below
            continue
        out.append(n)
    if not out and pool:                                   # every eligible block is taken: reuse the best one
        out.append(pool[0])
    return out[:k]


def batch_blocks(ctxs: list[dict], seed: int = 0, briefs: list | None = None, k: int = 2) -> list[list[str]]:
    """blocks for every piece of a batch: NO two pieces share a dominant block while enough eligible ones exist. Pieces with the fewest
    eligible blocks choose first; a brief that names a block gets it."""
    briefs = list(briefs or [None] * len(ctxs))
    order = sorted(range(len(ctxs)), key=lambda i: (len([n for n in NAMES if eligible(n, ctxs[i])[0]]), i))
    taken: set = set()
    out: list = [None] * len(ctxs)
    for i in order:
        got = choose_blocks(ctxs[i], seed, briefs[i], taken, tag=str(i), k=k)
        out[i] = got
        if got:
            taken.add(got[0])
    return out


def resolve_params(name: str, seed: int = 0, tag: str = "", overrides: dict | None = None) -> dict:
    """a value inside every parameter's range: the default for tag == "", else a seeded point of the range (ints rounded, choices picked)."""
    out = {}
    for k, v in (BLOCKS[name].get("params") or {}).items():
        if overrides and k in overrides:
            val = overrides[k]
            if v["kind"] == "choice":
                val = val if val in v["choices"] else v["default"]
            else:
                val = min(max(val, v["lo"]), v["hi"])
        elif not tag:
            val = v["default"]
        elif v["kind"] == "choice":
            val = v["choices"][_rank(seed, "param", tag, name, k) % len(v["choices"])]
        else:
            u = (_rank(seed, "param", tag, name, k) % 10007) / 10006.0
            val = v["lo"] + u * (v["hi"] - v["lo"])
            val = int(round(val)) if v["kind"] == "int" else round(val, 3)
        out[k] = val
    return out


def describe(names, seed: int = 0, n_lines: int = 5, tag: str = "p", overrides: dict | None = None) -> list[dict]:
    """what plan_rules stores in piece["plan"]["blocks"]: per block its status, resolved params, the lines it constrains, its compose templates
    and what is still missing. The first entry is the dominant block."""
    res = []
    for k, nm in enumerate(names):
        b = BLOCKS[nm]
        sl = b.get("slots") or {}
        ls = sl.get("lines") or {}
        res.append({"name": nm, "dominant": k == 0, "status": b["status"], "role": b["role"], "params": resolve_params(nm, seed, f"{tag}.{k}", (overrides or {}).get(nm)),
                    "lines": _which_lines(ls.get("which", "all"), n_lines) if ls else [],
                    "compose": [dict(c) for c in sl.get("compose", [])], "transitions": list(sl.get("transitions") or []),
                    "contour": dict(sl["contour"]) if sl.get("contour") else None, "shots": dict(sl["shots"]) if sl.get("shots") else None,
                    "image_transition": dict(sl["image_transition"]) if sl.get("image_transition") else None,
                    "missing": list(b.get("missing", []))})
    return res


def names_for_brief() -> list[tuple[str, str]]:
    """(name, one-line doc) pairs of the blocks a brief may name (spec blocks too: the planner reports their status)."""
    return [(n, BLOCKS[n]["doc"].split(":")[0].split(" (")[0][:110]) for n in NAMES]
