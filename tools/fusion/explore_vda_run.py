"""run Video Depth Anything (vits, the box checkpoint) on clips -> <out>/<name>.npy float16 (T,H,W) disparity (high = near), clip resolution."""
import sys, os, time, numpy as np, cv2, torch
sys.path.insert(0, "/workspace/mg_explore_cuts/vda_pkg")
from video_depth_anything.video_depth import VideoDepthAnything
class PB:
    def update(self, *a, **k): pass
m = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
sd = torch.load("/workspace/ComfyUI/models/videodepthanything/video_depth_anything_vits.pth", map_location="cpu")
m.load_state_dict(sd, strict=True); m = m.to("cuda").eval()
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for spec in sys.argv[2:]:
    name, path = spec.split("=", 1)
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f[..., ::-1])
    fr = np.stack(fr); t = time.time()
    d = m.infer_video_depth(fr, input_size=518, device="cuda", pbar=PB(), fp32=False)
    d = d[0] if isinstance(d, tuple) else d
    d = np.stack([cv2.resize(np.asarray(x, np.float32), (fr.shape[2], fr.shape[1]), interpolation=cv2.INTER_CUBIC) for x in d]).astype(np.float16)
    np.save(os.path.join(out, name + ".npy"), d)
    print(name, fr.shape, d.shape, float(d.min()), float(d.max()), round(time.time() - t, 1), "s", flush=True)
