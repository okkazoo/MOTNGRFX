"""Same-subject new-angle stills (tools/fusion/angle_still.py, idea 23_moss-angles): the anchored edit prompt, the one-angle graph with a seed list, and the
consistency pick. Pure: no model, no GPU, no ComfyUI; the pick is checked on synthetic images.
Run: python -m pytest tests/test_angle_still.py -q"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.fusion import angle_still as A  # noqa: E402

STILL = ("A bright red kayak glides across a glassy lake at sunrise, warm golden light reflecting off the calm water, morning mist rising gently from the surface, "
         "the distant shoreline silhouetted in soft focus, shot from a low mid-range angle, cinematic 35mm film still, shallow depth of field, tall vertical frame")


def test_split_still_prompt_separates_subject_light_and_style():
    p = A.split_still_prompt(STILL)
    assert p["subject"].startswith("A bright red kayak")
    assert any("golden light" in c for c in p["light"]) and any("mist" in c for c in p["light"])
    assert "the distant shoreline silhouetted in soft focus" in p["rest"]
    assert "shot from a low mid-range angle" in p["dropped"] and "cinematic 35mm film still" in p["dropped"] and "tall vertical frame" in p["dropped"]


@pytest.mark.parametrize("word,key", [("close-up", "close-up"), ("Close up", "close-up"), ("low angle", "low"), ("Low_Angle", "low"), ("overhead", "top-down"),
                                      ("top down", "top-down"), ("side view", "profile"), ("close-up shot", "close-up"), ("OTS", "over-the-shoulder")])
def test_framing_aliases(word, key):
    assert A.normalize_framing(word) == key


def test_unknown_framing_is_used_verbatim():
    assert A.normalize_framing("dutch tilt from the left") == "dutch tilt from the left"
    assert "dutch tilt from the left" in A.build_edit_prompt(STILL, "dutch tilt from the left")


def test_anchored_prompt_has_the_v002_parts():
    t = A.build_edit_prompt(STILL, "profile")
    assert t.startswith("Change only the camera:") and "side view in profile" in t
    assert "bright red kayak" in t and "warm golden light" in t                      # subject + palette restated from the still prompt
    assert "keep the same colours, light and mood" in t
    assert "film still" not in t and "mid-range angle" not in t                      # the still's own framing / style words are dropped
    assert A.FACE_HIDDEN not in t                                                    # profile: no face line by default
    assert A.SECOND_REF not in t


def test_subject_override_and_keep_and_palette():
    t = A.build_edit_prompt(STILL, "top-down", subject="paddler in a dark top and red life vest, in the red kayak", keep="dark-red paddle blades", palette="cool pale-blue dawn light")
    assert "Show the same paddler in a dark top and red life vest, in the red kayak, seen in the reference image, with dark-red paddle blades" in t
    assert "the same cool pale-blue dawn light" in t and "golden" not in t


def test_face_hidden_default_follows_the_framing_and_can_be_forced():
    assert A.FACE_HIDDEN in A.build_edit_prompt(STILL, "close-up")
    assert A.FACE_HIDDEN in A.build_edit_prompt(STILL, "low angle")
    assert A.FACE_HIDDEN not in A.build_edit_prompt(STILL, "top-down")
    assert A.FACE_HIDDEN not in A.build_edit_prompt(STILL, "close-up", hide_face=False)
    assert A.FACE_HIDDEN in A.build_edit_prompt(STILL, "profile", hide_face=True)
    assert A.FACE_HIDDEN not in A.build_edit_prompt(STILL, "close-up", person=False)             # not a person: no face line unless forced


def test_second_ref_line():
    assert A.SECOND_REF in A.build_edit_prompt(STILL, "low", second_ref=True)


def test_graph_one_branch_per_seed_with_shared_encoders():
    g = A.angle_graph("ref.png", "PROMPT", [1, 2, 3], prefix="x_close-up", size=(448, 832))
    cls = [n["class_type"] for n in g.values()]
    assert cls.count("KSampler") == 3 and cls.count("VAEDecode") == 3 and cls.count("SaveImage") == 3
    assert cls.count("TextEncodeMOSSImageEdit") == 2 and cls.count("UNETLoader") == 1 and cls.count("LoraLoaderModelOnly") == 1 and cls.count("LoadImage") == 1
    pos, neg = g["pos"]["inputs"], g["neg"]["inputs"]
    assert pos["edit_format"] == "Moss-Image-Edit" and pos["prompt"] == "PROMPT" and pos["vae"] == ["vae", 0]
    assert pos["images.image_0"] == ["ref0", 0] and neg["images.image_0"] == ["ref0", 0]          # the same reference for positive and negative
    assert "images.image_1" not in pos
    assert g["lora"]["inputs"]["lora_name"] == A.LORA and g["lora"]["inputs"]["model"] == ["unet", 0]
    ks = [g[f"ks_{i:02d}"]["inputs"] for i in range(3)]
    assert [k["seed"] for k in ks] == [1, 2, 3]
    assert all(k["model"] == ["lora", 0] and k["positive"] == ["pos", 0] and k["negative"] == ["neg", 0] for k in ks)
    assert all(k["steps"] == 30 and k["cfg"] == 4.0 and k["sampler_name"] == "er_sde" and k["scheduler"] == "simple" for k in ks)
    assert g["latent"]["inputs"]["width"] == 448 and g["latent"]["inputs"]["height"] == 832
    ids = A.seed_save_ids([1, 2, 3])
    assert ids == sorted(ids) and [g[i]["inputs"]["filename_prefix"] for i in ids] == ["x_close-up_s1", "x_close-up_s2", "x_close-up_s3"]


def test_graph_second_reference_and_links():
    g = A.angle_graph("orig.png", "P", 7, prefix="p", second_ref="closeup.png")
    assert g["pos"]["inputs"]["images.image_0"] == ["ref0", 0] and g["pos"]["inputs"]["images.image_1"] == ["ref1", 0]
    assert g["ref0"]["inputs"]["image"] == "orig.png" and g["ref1"]["inputs"]["image"] == "closeup.png"
    # existing IMAGE output as the reference (the app's decoded place still), existing loaders, merged ids
    g2 = A.angle_graph(["still_dec", 0], "P", [5], prefix="p", loaders={"unet": ["u", 0], "clip": ["c", 0], "vae": ["v", 0]}, id_prefix="ang_")
    cls = [n["class_type"] for n in g2.values()]
    assert "LoadImage" not in cls and "UNETLoader" not in cls and "VAELoader" not in cls and "CLIPLoader" not in cls
    assert g2["ang_pos"]["inputs"]["images.image_0"] == ["still_dec", 0] and g2["ang_pos"]["inputs"]["clip"] == ["c", 0]
    assert g2["ang_lora"]["inputs"]["model"] == ["u", 0] and all(k.startswith("ang_") for k in g2)
    assert A.seed_save_ids([5], "ang_") == ["ang_save_00"]


def test_graph_needs_a_seed():
    with pytest.raises(ValueError):
        A.angle_graph("r.png", "P", [], prefix="p")


def test_seed_list_distinct_and_first_is_the_base():
    s = A.seed_list(42, 4)
    assert s[0] == 42 and len(set(s)) == 4 and all(0 <= x < 2**31 for x in s)
    assert A.seed_list(42, 1) == [42] and A.seed_list(42, 4) == s


# ------------------------------------------------------------------------------------------------------------------------------------------------ the pick
def _scene(sky, water, subject=None, box=(40, 60, 60, 80), size=(160, 100)):
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    img[: h // 2] = sky
    img[h // 2:] = water
    if subject is not None:
        y0, x0, y1, x1 = box
        img[y0:y1, x0:x1] = subject
    return img


def test_score_identity_is_the_maximum_and_in_range():
    ref = _scene((240, 190, 120), (30, 40, 60), (200, 20, 30))
    s = A.consistency_score(ref, ref)
    assert s == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= A.consistency_score(ref, _scene((20, 20, 20), (240, 240, 240), (0, 200, 0))) < 0.5


def test_pick_prefers_the_same_light_and_clothes_over_a_colour_shift():
    ref = _scene((240, 190, 120), (30, 40, 60), (200, 20, 30))
    same = _scene((235, 185, 118), (32, 42, 62), (198, 22, 32), box=(30, 50, 70, 100))                # same grade + red vest, a different framing
    blue = _scene((150, 190, 235), (20, 40, 70), (200, 20, 30))                                         # the golden light went cool blue (the v002 drift)
    green = _scene((240, 190, 120), (30, 40, 60), (20, 200, 40))                                        # another vest colour
    idx, scores = A.pick_best(ref, [blue, green, same])
    assert idx == 2 and scores[2] > scores[0] and scores[2] > scores[1]


def test_pick_ties_take_the_earlier_seed_and_accept_float_images_and_masks():
    ref = _scene((240, 190, 120), (30, 40, 60), (200, 20, 30))
    idx, scores = A.pick_best(ref.astype(np.float32) / 255.0, [ref, ref.copy()])
    assert idx == 0 and scores[0] == pytest.approx(scores[1])
    m = np.zeros(ref.shape[:2], bool)
    m[40:60, 60:80] = True
    assert A.consistency_score(ref, ref, ref_mask=m, cand_mask=m) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError):
        A.pick_best(ref, [])
