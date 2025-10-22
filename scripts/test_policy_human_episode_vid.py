"""
USAGE:
uv run scripts/test_policy_human_episode_vid.py
# then stitch:
# ffmpeg -y -framerate 30 -i vis_human_ep/%06d.jpg -c:v libx264 -pix_fmt yuv420p vis_human_ep.mp4
"""

import os, glob, cv2, h5py, numpy as np
import jax.numpy as jnp

# ---------- config you edit ----------
ROOT_DIR   = "/iris/projects/humanoid/dataset/ego_dex"
# Option A) set exact episode files:
EP_H5      = "/iris/projects/humanoid/dataset/ego_dex/part3/make_sandwich/20.hdf5"     # or leave "" to auto-find by TASK_NAME + index
EP_MP4     = "/iris/projects/humanoid/dataset/ego_dex/part3/make_sandwich/20.mp4"      # must match the h5
# Option B) auto-pick by task name + episode index within that task
TASK_NAME  = "make_sandwich"             # folder name under a part/ dir
EP_INDEX   = 0                           # 0-based within that task
# Inference/render settings
IMG_SIZE   = 224
STRIDE     = 1                           # frame step for GT window building
PROMPT     = "make_sandwich"
OUT_DIR    = "vis_human_policy_make_sandwich"  # output per-frame images here
os.makedirs(OUT_DIR, exist_ok=True)

# Camera intrinsics (use your real K)
K = np.array([[736.6339, 0., 960.],
              [0., 736.6339, 540.],
              [0., 0., 1.]], dtype=np.float32)

# ---------- tiny helpers ----------
def _inv(T):
    Rm = T[:3,:3]; t = T[:3,3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3,:3] = Rm.T
    inv[:3,3]  = -Rm.T @ t
    return inv

def _rotmat_to_rot6d(Rm):
    return np.array([Rm[0,0],Rm[1,0],Rm[2,0],Rm[0,1],Rm[1,1],Rm[2,1]], dtype=np.float32)

def _pose_world_to_cam(T_world_obj, T_world_cam):
    return _inv(T_world_cam) @ T_world_obj

def _project(xyz, K):
    uv = np.full((len(xyz),2), np.nan, np.float32)
    Z = xyz[:,2]; m = Z > 1e-6
    uv[m,0] = K[0,0]*(xyz[m,0]/Z[m]) + K[0,2]
    uv[m,1] = K[1,1]*(xyz[m,1]/Z[m]) + K[1,2]
    return uv

def _draw_path(img_bgr, uv, color, label=None):
    pts = [(int(round(u)), int(round(v))) for u,v in uv if np.isfinite(u) and np.isfinite(v)]
    if len(pts) >= 2:
        cv2.polylines(img_bgr, [np.array(pts, np.int32)], False, color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img_bgr, p, 2, color, -1, cv2.LINE_AA)
    if label and pts:
        cv2.putText(img_bgr, label, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

def _read_rgb_resized(mp4_path, t, wh=(IMG_SIZE, IMG_SIZE)):
    W,H = wh
    for _ in range(3):
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            cap.release(); continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, bgr = cap.read(); cap.release()
        if ok and bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (W,H), interpolation=cv2.INTER_AREA)
            return (rgb.astype(np.float32)/255.0)
    return np.zeros((H,W,3), np.float32)

def _read_rgb_orig(mp4_path, t):
    for _ in range(3):
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            cap.release(); continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, bgr = cap.read(); cap.release()
        if ok and bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return (rgb.astype(np.float32)/255.0)
    return None

# ---------- state builders (match your EgoDexSeqDataset "ego_split") ----------
WRISTS = ["leftHand","rightHand"]
FINGERTIPS = [
    "leftThumbTip","leftIndexFingerTip","leftMiddleFingerTip","leftRingFingerTip","leftLittleFingerTip",
    "rightThumbTip","rightIndexFingerTip","rightMiddleFingerTip","rightRingFingerTip","rightLittleFingerTip",
]

def _state_ego48(f, t):
    T_world_cam = f["transforms"]["camera"][t]
    ws = []
    for joint in WRISTS:
        T_world = f["transforms"][joint][t]
        T_cam   = _pose_world_to_cam(T_world, T_world_cam)
        pos  = T_cam[:3,3].astype(np.float32)
        rot6 = _rotmat_to_rot6d(T_cam[:3,:3])
        ws.append(np.concatenate([pos, rot6], dtype=np.float32))
    wrists_vec = np.concatenate(ws, dtype=np.float32)  # 18
    tips = []
    for joint in FINGERTIPS:
        T_world = f["transforms"][joint][t]
        T_cam   = _pose_world_to_cam(T_world, T_world_cam)
        tips.append(T_cam[:3,3].astype(np.float32))
    tips_vec = np.concatenate(tips, dtype=np.float32)  # 30
    return np.concatenate([wrists_vec, tips_vec], dtype=np.float32)  # 48

def _ego48_to_split32_tokens(seq48):
    """seq48: (F,48) -> tokens: (2F,32) with each 24-d half padded to 32-d."""
    first  = seq48[:, :24]
    second = seq48[:, 24:]
    tokens24 = np.empty((seq48.shape[0]*2, 24), np.float32)
    tokens24[0::2] = first
    tokens24[1::2] = second
    out = np.zeros((tokens24.shape[0], 32), np.float32)
    out[:, :24] = tokens24
    return out

# ---------- episode picker ----------
def _find_episode_by_task(root_dir, task_name, ep_index=0):
    parts = [d for d in sorted(os.listdir(root_dir))
             if os.path.isdir(os.path.join(root_dir,d)) and (d.startswith("part") or d in ("test","extra"))]
    for part in parts:
        task_dir = os.path.join(root_dir, part, task_name)
        if not os.path.isdir(task_dir): continue
        h5s = sorted(glob.glob(os.path.join(task_dir, "*.hdf5")))
        eps = []
        for h5 in h5s:
            mp4 = h5.replace(".hdf5", ".mp4")
            if os.path.exists(mp4): eps.append((h5, mp4))
        if eps:
            ep_index = max(0, min(ep_index, len(eps)-1))
            return eps[ep_index]
    return None, None

# ---------- policy ----------
from openpi.training import config as cfg
from openpi.policies import policy_config
from openpi.shared import download

def _load_policy():
    conf     = cfg.get_config("pi05_galaxea_egodex_pick_place")
    # change if you want a different checkpoint
    # ckpt_dir = download.maybe_download("checkpoints/pi05_mixed/my_experiment_co_training/35000")
    ckpt_dir  = download.maybe_download("/iris/projects/humanoid/openpi/checkpoints/pi05_galaxea_egodex_pick_place/galaxea_egodex_pick_place_delta_native/10000")
    return policy_config.create_trained_policy(conf, ckpt_dir)

# ---------- main ----------
def main():
    h5_path, mp4_path = EP_H5, EP_MP4
    if not (h5_path and mp4_path):
        h5_path, mp4_path = _find_episode_by_task(ROOT_DIR, TASK_NAME, EP_INDEX)
    if not (h5_path and mp4_path):
        raise RuntimeError("Could not locate episode. Set EP_H5/EP_MP4 or TASK_NAME/EP_INDEX correctly.")

    # load episode length
    with h5py.File(h5_path, "r") as f:
        N = int(f["transforms"]["leftHand"].shape[0])

    policy = _load_policy()

    print(f"Running episode:\n  H5 : {h5_path}\n  MP4: {mp4_path}\n  N  : {N}")
    for t in range(N):
        # print going through frame t out of n
        print(f"Processing frame {t+1}/{N}")
        # input image for the policy (224x224)
        img_infer = _read_rgb_resized(mp4_path, t, (IMG_SIZE, IMG_SIZE))
        if img_infer is None:
            print(f"[skip] frame {t}"); continue

        # build GT window to match predicted token length
        # first do a dry-run to know pred length (tokens)
        example = {
            "state": jnp.asarray(np.zeros((32,), np.float32)),  # dummy; replaced below after we compute real state
            "image": jnp.asarray(img_infer),
            "prompt": PROMPT,
        }
        # quick one-step infer just to get output length (many backends are fine with dummy state)
        pred_dummy = np.asarray(policy.infer(example)["actions"])
        T_pred_tokens = len(pred_dummy)              # e.g., 50
        F_frames = max(1, T_pred_tokens // 2)        # frames represented by those tokens

        # now compute real state + actions window from episode
        with h5py.File(h5_path, "r") as f:
            ts = [tt for tt in range(t, min(N, t + STRIDE*F_frames), STRIDE)]
            if len(ts) == 0:
                continue
            seq48 = np.stack([_state_ego48(f, tt) for tt in ts], axis=0)  # (F,48)
            actions_tokens32 = _ego48_to_split32_tokens(seq48)            # (2F,32)
            # trim/pad to T_pred_tokens so GT matches pred length
            L = min(T_pred_tokens, len(actions_tokens32))
            actions_tokens32 = actions_tokens32[:L]
            # state = first token
            state32 = actions_tokens32[0].copy()

        # final example for policy
        example = {
            "state": jnp.asarray(state32),
            "image": jnp.asarray(img_infer),
            "prompt": PROMPT,
        }
        pred = np.asarray(policy.infer(example)["actions"])   # (T_pred_tokens, 26)

        # extract L/R wrist xyz from pred tokens and GT tokens
        # (token layout: even=L, odd=R; xyz at [:3] for each)
        L_pred_xyz = pred[0::2, 0:3]
        R_pred_xyz = pred[1::2, 0:3]

        L_gt_xyz   = actions_tokens32[0::2, 0:3]
        R_gt_xyz   = actions_tokens32[0::2, 9:12]

        # overlay on original-resolution image
        img_orig = _read_rgb_orig(mp4_path, t)
        if img_orig is None:
            print(f"[skip overlay] frame {t}"); continue
        h, w = img_orig.shape[:2]
        img_draw = cv2.cvtColor((img_orig*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        Lp = _project(L_pred_xyz, K); Rp = _project(R_pred_xyz, K)
        Lg = _project(L_gt_xyz,   K); Rg = _project(R_gt_xyz,   K)

        _draw_path(img_draw, Lg, (0,200,0),   "L GT")
        _draw_path(img_draw, Lp, (0,255,255), "L Pred")
        _draw_path(img_draw, Rg, (0,0,200),   "R GT")
        _draw_path(img_draw, Rp, (255,0,255), "R Pred")

        # small legend box
        cv2.rectangle(img_draw, (10,10), (250,75), (255,255,255), -1)
        cv2.putText(img_draw, "L: GT / Pred", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,150,0),   2, cv2.LINE_AA)
        cv2.putText(img_draw, "R: GT / Pred", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,150),   2, cv2.LINE_AA)

        out_path = os.path.join(OUT_DIR, f"{t:06d}.jpg")
        cv2.imwrite(out_path, img_draw)

    print(f"Saved per-frame overlays to: {OUT_DIR}")

if __name__ == "__main__":
    main()
