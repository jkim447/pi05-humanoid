"""
USAGE:
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/test_policy_robot_episode_vid.py
"""

# ─────────────────────────── imports ────────────────────────────
import os
import cv2
import pandas as pd
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # kept import in case you want quick debugging plots
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from openpi.training import config as cfg
from openpi.policies import policy_config
from openpi.shared import download
from scipy.spatial.transform import Rotation as R

# ─────────────—— file locations & basic constants —──────────────
idx = 30
stride = 3
demo = 25
IMG_SIZE = 224
PROMPT   = "place_red_cube into box"

IMG_FMT = f"/iris/projects/humanoid/tesollo_dataset/robot_data_0903/red_cube_inbox/demo_{demo}/left/{{:06d}}.jpg"
CSV_PATH = f"/iris/projects/humanoid/tesollo_dataset/robot_data_0903/red_cube_inbox/demo_{demo}/ee_hand.csv"

right_hand_cols = [f"right_hand_{i}" for i in range(20)]

# Intrinsics (use as-is, no scaling)
K_LEFT  = np.array([[730.2571411132812, 0.0, 637.2598876953125],
                    [0.0, 730.2571411132812, 346.41082763671875],
                    [0.0, 0.0, 1.0]], dtype=np.float64)

# Extrinsics (camera frame to the robot frame (inverse of the camera calib extrinsics))
T_BASE_TO_CAM_LEFT = np.linalg.inv(np.array([
    [ 0.00692993, -0.87310148,  0.48748926,  0.14062141],
    [-0.99995006, -0.00956093, -0.00290894,  0.03612369],
    [ 0.00720065, -0.48744476, -0.87312414,  0.46063114],
    [ 0., 0., 0., 1. ]
], dtype=np.float64))

# ─────────────────────────── helpers ────────────────────────────
def _project_cam_to_px(cam_xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """cam_xyz: (N,3) in left camera frame; returns (N,2) pixel coords (u,v)."""
    X, Y, Z = cam_xyz[:, 0], cam_xyz[:, 1], cam_xyz[:, 2]
    eps = 1e-6
    Z = np.clip(Z, eps, None)
    u = K[0, 0] * (X / Z) + K[0, 2]
    v = K[1, 1] * (Y / Z) + K[1, 2]
    return np.stack([u, v], axis=1)

def _world_to_cam3(p_world3):
    """Transform a single 3D point from robot base -> left camera frame."""
    ph = np.array([p_world3[0], p_world3[1], p_world3[2], 1.0], dtype=np.float64)
    pc = T_BASE_TO_CAM_LEFT @ ph
    return pc[:3]  # (Xc, Yc, Zc)

def _rot_base_to_cam(R_base):
    return (T_BASE_TO_CAM_LEFT[:3,:3] @ R_base).astype(np.float32)

def _row_to_action(row):
    # Right wrist position in left camera frame
    p_world = np.array([row["right_pos_x"], row["right_pos_y"], row["right_pos_z"]], dtype=np.float64)
    p_cam = _world_to_cam3(p_world)

    # Right wrist orientation (quat -> 6D) in left camera frame
    rq = [row["right_ori_x"], row["right_ori_y"], row["right_ori_z"], row["right_ori_w"]]
    rR = R.from_quat(rq).as_matrix()
    R_cam = _rot_base_to_cam(rR)
    ori6d = R_cam[:, :2].reshape(-1, order="F")

    # Right hand command joints (20)
    joints20 = np.asarray([row[c] for c in right_hand_cols], dtype=np.float32)

    # Final action vector: [pos_cam(3), ori6d(6), joints20(20)] = 29 dims
    a = np.concatenate([p_cam.astype(np.float32), ori6d.astype(np.float32), joints20], axis=0)
    return a  # (29,)

def _draw_path(img_bgr, pts, color_bgr, thickness=2, radius=2):
    if len(pts) == 0:
        return
    for i in range(1, len(pts)):
        cv2.line(img_bgr, (int(pts[i-1,0]), int(pts[i-1,1])),
                           (int(pts[i,0]),   int(pts[i,1])), color_bgr, thickness, lineType=cv2.LINE_AA)
    for i in range(len(pts)):
        cv2.circle(img_bgr, (int(pts[i,0]), int(pts[i,1])), radius, color_bgr, -1, lineType=cv2.LINE_AA)

def _in_bounds(p, w, h):
    return (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)

# ──────────────────── load policy & warm up ─────────────────────
conf      = cfg.get_config("pi05_galaxea")
ckpt_dir  = download.maybe_download("checkpoints/pi05_galaxea/my_galaxea/28000")
policy    = policy_config.create_trained_policy(conf, ckpt_dir)

# ───────────────────────── episode loop ─────────────────────────
OUT_DIR = f"vis_demo_{demo}"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
N  = len(df)

for t in range(N):
    # ----- build per-frame inputs aligned to time t -----
    img_path_t = IMG_FMT.format(t)

    # image for inference (224×224 like training)
    img_infer = cv2.imread(img_path_t)
    if img_infer is None:
        print(f"[skip] cannot read {img_path_t}")
        continue
    img_infer = cv2.cvtColor(img_infer, cv2.COLOR_BGR2RGB)
    img_infer = cv2.resize(img_infer, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img_infer = img_infer.astype(np.float32) / 255.0

    # state at time t (29 -> pad to 32)
    row_t   = df.iloc[t]
    state29 = _row_to_action(row_t).astype(np.float32)   # (29,)
    state32 = np.pad(state29, (0, 3)).astype(np.float32) # (32,)

    example = {
        "state": jnp.asarray(state32),
        "image": jnp.asarray(img_infer),
        "prompt": PROMPT,
    }

    # ----- inference -----
    pred = np.asarray(policy.infer(example)["actions"])   # e.g., (T_pred, 26)
    traj_pred_R = pred[1::2, 0:3]                         # right-wrist cam-frame XYZ

    # ----- build GT window starting at t, matching pred length/stride -----
    T_half = int(len(pred) // 2)
    gt_rows = df.iloc[t : t + stride * T_half : stride]
    if len(gt_rows) == 0:
        continue
    gt = np.stack([_row_to_action(r) for _, r in gt_rows.iterrows()], axis=0)  # (L, 29)
    traj_gt_R = gt[:, 0:3]

    # trim to common length
    L = min(len(traj_pred_R), len(traj_gt_R))
    traj_pred_R = traj_pred_R[:L]
    traj_gt_R   = traj_gt_R[:L]

    # ----- overlay on original-resolution image (use original K_LEFT) -----
    img_draw = cv2.imread(img_path_t)  # BGR original size
    if img_draw is None:
        print(f"[skip overlay] cannot read {img_path_t}")
        continue
    h, w = img_draw.shape[:2]

    px_pred = _project_cam_to_px(traj_pred_R, K_LEFT)
    px_gt   = _project_cam_to_px(traj_gt_R,   K_LEFT)

    m_pred = _in_bounds(px_pred, w, h)
    m_gt   = _in_bounds(px_gt,   w, h)
    px_pred = px_pred[m_pred].astype(np.int32)
    px_gt   = px_gt[m_gt].astype(np.int32)

    _DRAW_RED   = (0,   0, 255)  # Pred
    _DRAW_GREEN = (0, 255,   0)  # GT
    _draw_path(img_draw, px_gt,   _DRAW_GREEN, thickness=2, radius=2)
    _draw_path(img_draw, px_pred, _DRAW_RED,   thickness=2, radius=2)

    # tiny legend
    cv2.rectangle(img_draw, (10, 10), (260, 70), (255, 255, 255), -1)
    cv2.putText(img_draw, "GT right wrist",   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _DRAW_GREEN, 2, cv2.LINE_AA)
    cv2.putText(img_draw, "Pred right wrist", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _DRAW_RED,   2, cv2.LINE_AA)

    # save per-frame
    out_path = os.path.join(OUT_DIR, f"{t:06d}.jpg")
    cv2.imwrite(out_path, img_draw)

print(f"Saved per-frame full trajectories to: {OUT_DIR}")

# (optional) quick ffmpeg stitch (run in shell):
# ffmpeg -y -framerate 30 -i vis_demo_{demo}/%06d.jpg -c:v libx264 -pix_fmt yuv420p vis_demo_{demo}.mp4
