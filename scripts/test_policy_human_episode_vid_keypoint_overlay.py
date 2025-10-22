"""
USAGE:
uv run scripts/test_policy_human_episode_vid_keypoint_overlay.py
# then stitch:
# ffmpeg -y -framerate 30 -i vis_human_ep/%06d.jpg -c:v libx264 -pix_fmt yuv420p vis_human_ep.mp4
"""

import os, glob, cv2, h5py, numpy as np
import jax.numpy as jnp

# ---------- config you edit ----------
ROOT_DIR   = "/iris/projects/humanoid/dataset/ego_dex"
# Option A) set exact episode files:
EP_H5      = "/iris/projects/humanoid/dataset/ego_dex/part4/setup_cleanup_table/105.hdf5"     # or leave "" to auto-find by TASK_NAME + index
EP_MP4     = "/iris/projects/humanoid/dataset/ego_dex/part4/setup_cleanup_table/105.mp4"      # must match the h5
# Option B) auto-pick by task name + episode index within that task
TASK_NAME  = "setup_cleanup_table"             # folder name under a part/ dir
EP_INDEX   = 0                           # 0-based within that task
# Inference/render settings
IMG_SIZE   = 224
STRIDE     = 1                           # frame step for GT window building
PROMPT     = "setup_cleanup_table"
OUT_DIR    = "vis_human_policy_setup_cleanup_table"  # output per-frame images here
os.makedirs(OUT_DIR, exist_ok=True)

from dataclasses import dataclass
from openpi.training.hand_keypoints_config import LEFT_FINGERS, RIGHT_FINGERS, JOINT_COLOR_BGR

# 21-joint lists (order matches your dataset)
LEFT_MANO_21 = [
    "leftThumbKnuckle","leftThumbIntermediateBase","leftThumbIntermediateTip","leftThumbTip",
    "leftIndexFingerKnuckle","leftIndexFingerIntermediateBase","leftIndexFingerIntermediateTip","leftIndexFingerTip",
    "leftMiddleFingerKnuckle","leftMiddleFingerIntermediateBase","leftMiddleFingerIntermediateTip","leftMiddleFingerTip",
    "leftRingFingerKnuckle","leftRingFingerIntermediateBase","leftRingFingerIntermediateTip","leftRingFingerTip",
    "leftLittleFingerKnuckle","leftLittleFingerIntermediateBase","leftLittleFingerIntermediateTip","leftLittleFingerTip",
]
RIGHT_MANO_21 = [
    "rightThumbKnuckle","rightThumbIntermediateBase","rightThumbIntermediateTip","rightThumbTip",
    "rightIndexFingerKnuckle","rightIndexFingerIntermediateBase","rightIndexFingerIntermediateTip","rightIndexFingerTip",
    "rightMiddleFingerKnuckle","rightMiddleFingerIntermediateBase","rightMiddleFingerIntermediateTip","rightMiddleFingerTip",
    "rightRingFingerKnuckle","rightRingFingerIntermediateBase","rightRingFingerIntermediateTip","rightRingFingerTip",
    "rightLittleFingerKnuckle","rightLittleFingerIntermediateBase","rightLittleFingerIntermediateTip","rightLittleFingerTip",
]



# Camera intrinsics (use your real K)
K = np.array([[736.6339, 0., 960.],
              [0., 736.6339, 540.],
              [0., 0., 1.]], dtype=np.float32)

# ---------- tiny helpers ----------
def _project_pts_onto_resized(pts_cam_xyz, K, W, H):
    """Project Nx3 camera-frame points onto an image resized from 1920x1080 -> (W,H)."""
    W0, H0 = 1920.0, 1080.0
    sx, sy = W / W0, H / H0
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    uv = []
    for x, y, z in pts_cam_xyz:
        if z <= 1e-6:
            uv.append(None); continue
        u = (fx * x / z) + cx
        v = (fy * y / z) + cy
        uv.append((int(round(u * sx)), int(round(v * sy))))
    return uv

def draw_skeleton_occlusion_aware(image_rgb_float, names, uv, z, edges_by_name, color_of,
                                  pt_radius=8, line_thickness=4, edge_segments=12):
    """Depth-sort edge segments + points (far→near) and draw."""
    H, W = image_rgb_float.shape[:2]
    name_to_idx = {n:i for i,n in enumerate(names)}
    prims = []
    # Edges
    for a, b in edges_by_name:
        if a not in name_to_idx or b not in name_to_idx: 
            continue
        ia, ib = name_to_idx[a], name_to_idx[b]
        pa, pb = uv[ia], uv[ib]
        if (pa is None) or (pb is None): 
            continue
        ua, va = pa; ub, vb = pb
        za, zb = float(z[ia]), float(z[ib])
        for k in range(edge_segments):
            t0 = k / edge_segments; t1 = (k+1) / edge_segments
            u0 = int(round(ua*(1-t0) + ub*t0)); v0 = int(round(va*(1-t0) + vb*t0))
            u1 = int(round(ua*(1-t1) + ub*t1)); v1 = int(round(va*(1-t1) + vb*t1))
            if not (0 <= u0 < W and 0 <= v0 < H and 0 <= u1 < W and 0 <= v1 < H):
                continue
            zmid = za*(1-(t0+t1)/2) + zb*((t0+t1)/2)
            col  = color_of.get(b, (210,210,210))   # distal-node color
            prims.append(("edge", zmid, (u0,v0), (u1,v1), col))
    # Points
    for i, p in enumerate(uv):
        if p is None: 
            continue
        u, v = p
        if 0 <= u < W and 0 <= v < H:
            prims.append(("pt", float(z[i]), (u,v), color_of.get(names[i], (210,210,210))))

    prims.sort(key=lambda x: -x[1])  # far→near
    img_bgr = cv2.cvtColor((image_rgb_float*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    for prim in prims:
        if prim[0] == "edge":
            _, _, p0, p1, col = prim
            cv2.line(img_bgr, p0, p1, col, line_thickness, cv2.LINE_AA)
        else:
            _, _, (u,v), col = prim
            cv2.circle(img_bgr, (u, v), pt_radius, col, -1, lineType=cv2.LINE_AA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    return img_rgb

def _pose_cam_pos6(f, name, t):
    """Return (pos3, rot6) of a transform in camera frame at time t."""
    T_world_cam = f["transforms"]["camera"][t]
    T_world_obj = f["transforms"][name][t]
    T_cam = _pose_world_to_cam(T_world_obj, T_world_cam)
    pos  = T_cam[:3, 3].astype(np.float32)
    rot6 = _rotmat_to_rot6d(T_cam[:3, :3])
    return pos, rot6

def _pose_cam_pos3(f, name, t):
    if name not in f["transforms"]:
        return np.zeros(3, np.float32)
    T_world_cam = f["transforms"]["camera"][t]
    T_world_obj = f["transforms"][name][t]
    T_cam = _pose_world_to_cam(T_world_obj, T_world_cam)
    return T_cam[:3, 3].astype(np.float32)

def _hand_action24(f, t, side):
    """(24,) = wrist pos3+rot6 + 5 fingertips xyz (thumb→index→middle→ring→little)."""
    wrist = "leftHand" if side == "left" else "rightHand"
    pos, rot6 = _pose_cam_pos6(f, wrist, t)
    tips_order = ["Thumb","Index","Middle","Ring","Little"]
    names = [f"{side}{n}FingerTip" if n!="Thumb" else f"{side}{n}Tip" for n in tips_order]
    tips = [ _pose_cam_pos3(f, n, t) for n in names ]
    return np.concatenate([pos, rot6, *tips], dtype=np.float32)

def _state_both30(f, t):
    """(30,) = L(pos3+rot6) + R(pos3+rot6) + L_thumb + L_index + R_thumb + R_index."""
    Lp, Lr = _pose_cam_pos6(f, "leftHand",  t)
    Rp, Rr = _pose_cam_pos6(f, "rightHand", t)
    L_thumb = _pose_cam_pos3(f, "leftThumbTip",       t)
    L_index = _pose_cam_pos3(f, "leftIndexFingerTip", t)
    R_thumb = _pose_cam_pos3(f, "rightThumbTip",      t)
    R_index = _pose_cam_pos3(f, "rightIndexFingerTip",t)
    return np.concatenate([Lp, Lr, Rp, Rr, L_thumb, L_index, R_thumb, R_index], dtype=np.float32)

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

    # episode length
    with h5py.File(h5_path, "r") as f:
        N = int(f["transforms"]["leftHand"].shape[0])

    policy = _load_policy()

    print(f"Running episode:\n  H5 : {h5_path}\n  MP4: {mp4_path}\n  N  : {N}")
    for t in range(N):
        print(f"Processing frame {t+1}/{N}")

        # ------- quick dummy pass to get token horizon -------
        img0 = _read_rgb_orig(mp4_path, t)
        if img0 is None:
            print(f"[skip] frame {t}"); 
            continue
        img0_resized = cv2.resize(img0, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
        dummy_ex = {
            "state": jnp.asarray(np.zeros((30,), np.float32)),  # 30-D state
            "image": jnp.asarray(img0_resized),
            "prompt": PROMPT,
        }
        pred_dummy = np.asarray(policy.infer(dummy_ex)["actions"])
        T_pred_tokens = int(len(pred_dummy))            # e.g., 50
        F_frames = max(1, T_pred_tokens // 2)          # L,R tokens → frames

        # ------- build GT state (30) + relative actions (2F,24) -------
        with h5py.File(h5_path, "r") as f:
            ts = [tt for tt in range(t, min(N, t + STRIDE * F_frames), STRIDE)]
            if not ts:
                continue

            # base at t0 (for_hand_action24 relative deltas)
            base_L = _hand_action24(f, ts[0], "left")
            base_R = _hand_action24(f, ts[0], "right")

            actions_lr = []  # [L24, R24, L24, R24, ...]
            for tt in ts:
                aL = np.asarray(_hand_action24(f, tt, "left"),  dtype=np.float32)  - base_L.astype(np.float32)
                aR = np.asarray(_hand_action24(f, tt, "right"), dtype=np.float32)  - base_R.astype(np.float32)
                actions_lr.append(aL)
                actions_lr.append(aR)
            actions_tokens24 = np.stack(actions_lr, axis=0).astype(np.float32)  # (2F,24)

            # trim to match model output length
            L_tok = min(T_pred_tokens, len(actions_tokens24))
            actions_tokens24 = actions_tokens24[:L_tok]

            # state at start (30-D)
            state30 = _state_both30(f, ts[0]).astype(np.float32)

        # ------- build occlusion-aware overlay image (policy input) -------
        # gather joints for overlay at time t
        with h5py.File(h5_path, "r") as f:
            T_world_cam = f["transforms"]["camera"][t]
            names = (
                ["leftHand"] + [n for n in LEFT_MANO_21 if n in f["transforms"]] +
                ["rightHand"] + [n for n in RIGHT_MANO_21 if n in f["transforms"]]
            )
            names = [n for n in names if n in f["transforms"]]

            pts_cam = []
            for n in names:
                T_w = f["transforms"][n][t]
                T_c = _pose_world_to_cam(T_w, T_world_cam)
                pts_cam.append(T_c[:3, 3])
            pts_cam = np.asarray(pts_cam, dtype=np.float64)

        H0, W0 = img0.shape[:2]
        uv = _project_pts_onto_resized(pts_cam, K, W0, H0)
        z  = pts_cam[:, 2].astype(np.float32)

        # edges (wrist→knuckle + finger chains)
        edges_by_name = []
        def _add_chain(chain):
            for u, v in zip(chain, chain[1:]):
                if (u in names) and (v in names):
                    edges_by_name.append((u, v))
        if "leftHand" in names:
            for chain in LEFT_FINGERS:
                if chain and (chain[0] in names):
                    edges_by_name.append(("leftHand", chain[0]))
                _add_chain(chain)
        if "rightHand" in names:
            for chain in RIGHT_FINGERS:
                if chain and (chain[0] in names):
                    edges_by_name.append(("rightHand", chain[0]))
                _add_chain(chain)

        color_of = {n: JOINT_COLOR_BGR.get(n, (210,210,210)) for n in names}
        img_overlay = draw_skeleton_occlusion_aware(
            image_rgb_float=img0.astype(np.float32),
            names=names, uv=uv, z=z,
            edges_by_name=edges_by_name, color_of=color_of,
            pt_radius=10, line_thickness=6, edge_segments=12,
        )
        img_infer = cv2.resize(img_overlay, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)

        # ------- policy inference on overlaid input -------
        example = {"state": jnp.asarray(state30), "image": jnp.asarray(img_infer), "prompt": PROMPT}
        pred = np.asarray(policy.infer(example)["actions"])   # (T_pred_tokens, 24) alternating L,R

        # ------- reconstruct absolute GT + draw paths -------
        # predicted wrist xyz from tokens
        L_pred_xyz = base_L[0:3] + pred[0::2, 0:3]
        R_pred_xyz = base_R[0:3] + pred[1::2, 0:3]

        # absolute GT from base + deltas
        gt_tokens_abs = []
        for i in range(len(actions_tokens24)):
            if i % 2 == 0:   # left
                gt_tokens_abs.append(base_L + actions_tokens24[i])
            else:            # right
                gt_tokens_abs.append(base_R + actions_tokens24[i])
        gt_tokens_abs = np.stack(gt_tokens_abs, axis=0) if gt_tokens_abs else np.zeros((0,24), np.float32)

        L_gt_xyz = gt_tokens_abs[0::2, 0:3] if len(gt_tokens_abs) else np.zeros((0,3), np.float32)
        R_gt_xyz = gt_tokens_abs[1::2, 0:3] if len(gt_tokens_abs) else np.zeros((0,3), np.float32)

        # overlay wrist paths onto original-res frame
        img_draw = cv2.cvtColor((img0*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        Lp = _project(L_pred_xyz, K); Rp = _project(R_pred_xyz, K)
        Lg = _project(L_gt_xyz,   K); Rg = _project(R_gt_xyz,   K)

        _draw_path(img_draw, Lg, (0,200,0),   "L GT")
        _draw_path(img_draw, Lp, (0,255,255), "L Pred")
        _draw_path(img_draw, Rg, (0,0,200),   "R GT")
        _draw_path(img_draw, Rp, (255,0,255), "R Pred")

        # legend
        cv2.rectangle(img_draw, (10,10), (250,75), (255,255,255), -1)
        cv2.putText(img_draw, "L: GT / Pred", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,150,0), 2, cv2.LINE_AA)
        cv2.putText(img_draw, "R: GT / Pred", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,150), 2, cv2.LINE_AA)

        out_path = os.path.join(OUT_DIR, f"{t:06d}.jpg")
        cv2.imwrite(out_path, img_draw)

    print(f"Saved per-frame overlays to: {OUT_DIR}")


if __name__ == "__main__":
    main()
