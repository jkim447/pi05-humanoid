"""
USAGE: 
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/test_robot_policy_simple_robot.py

Galaxea-pi0 inference + 3-D comparison plot
──────────────────────────────────────────
• Uses the *real* proprio state from the same CSV row as the image.
• Plots wrist-position trajectories (left: dims 0-2, right: dims 7-9)
  for both prediction and ground truth, then saves to disk.
• Adjust paths, image resize size, and _STATE_COLS list if you changed them.
"""

# ─────────────────────────── imports ────────────────────────────
import cv2
import pandas as pd
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from openpi.training import config as cfg
from openpi.policies import policy_config
from openpi.shared import download
from scipy.spatial.transform import Rotation as R


# ─────────────—— file locations & basic constants —──────────────
idx = 0
stride = 3
IMG_PATH = f"/iris/projects/humanoid/tesollo_dataset/robot_data_0903/red_cube_inbox/demo_1/left/000000.jpg"
CSV_PATH = f"/iris/projects/humanoid/tesollo_dataset/robot_data_0903/red_cube_inbox/demo_1/ee_hand.csv"
PROMPT   = "place_red_cube into box"
IMG_SIZE = 224              # use 224 if that’s what you trained on

right_hand_cols = [f"right_hand_{i}" for i in range(20)]

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


def _world_to_cam3(p_world3):
    """Transform a single 3D point from robot base -> left camera frame."""
    ph = np.array([p_world3[0], p_world3[1], p_world3[2], 1.0], dtype=np.float64)
    pc = T_BASE_TO_CAM_LEFT @ ph
    return pc[:3]  # (Xc, Yc, Zc)

def _rot_base_to_cam(R_base):
    return (T_BASE_TO_CAM_LEFT[:3,:3] @ R_base).astype(np.float32)


def _row_to_action(row):
    # --- Right wrist position in left camera frame ---
    p_world = np.array([row["right_pos_x"], row["right_pos_y"], row["right_pos_z"]], dtype=np.float64)
    p_cam = _world_to_cam3(p_world)  # (3,)

    # --- Right wrist orientation in left camera frame  ---
    rq = [row["right_ori_x"], row["right_ori_y"], row["right_ori_z"], row["right_ori_w"]]
    rR = R.from_quat(rq).as_matrix()
    R_cam = _rot_base_to_cam(rR)
    ori6d = R_cam[:, :2].reshape(-1, order="F")

    # --- Right hand command joints (20) ---
    joints20 = np.asarray([row[c] for c in right_hand_cols], dtype=np.float32)

    # --- Final action vector: [pos_cam(3), ori6d(6), joints20(20)] = 29 dims ---
    a = np.concatenate([p_cam.astype(np.float32), ori6d.astype(np.float32), joints20], axis=0)
    return a  # (29,)

# ─────────────────────── load image ─────────────────────────────
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Cannot read {IMG_PATH}")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
img = img.astype(np.float32) / 255.0  # normalize to 0–1

# ────────────────────── load CSV & state ────────────────────────
df   = pd.read_csv(CSV_PATH)
row0 = df.iloc[0]               # take first frame; change if needed

# Build 26-D state vector in the *same order* used during training
state = _row_to_action(row0).astype(np.float32)

# ─────────────── build inference example dict ───────────────────
example = {
    "state": state,
    "image": img,
    "prompt": PROMPT,
}
example = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
           for k, v in example.items()}

# ──────────────────── load policy & infer ───────────────────────
conf      = cfg.get_config("pi05_galaxea")
ckpt_dir  = download.maybe_download("checkpoints/pi05_galaxea/my_galaxea/28000")
policy    = policy_config.create_trained_policy(conf, ckpt_dir)

pred = np.asarray(policy.infer(example)["actions"])   # shape (T, 26)

# ───────────────────── get matching ground truth ─────────────────
# The CSV is 30 fps; model actions are per 2 frames in your earlier code,
# so we sub-sample every 2nd row to align lengths.
gt_rows = df.iloc[idx : idx + stride * int(len(pred)//2) : stride]
gt = np.stack([_row_to_action(row) for _, row in gt_rows.iterrows()], axis=0)

# ───────────────────── slice wrist trajectories ─────────────────
# left wrist xyz  = dims 0-2, right wrist xyz = dims 7-9
# traj_pred_L = pred[:, 0:3]
traj_pred_R = pred[1::2, 0:3]

# traj_gt_L   = gt[:, 0:3]
traj_gt_R   = gt[1::2, 0:3]

# ─────────────────────────── plot & save ────────────────────────
fig = plt.figure()
ax  = fig.add_subplot(111, projection="3d")

ax.plot(*traj_pred_R.T, label="pred right wrist", color="tab:red")
ax.plot(*traj_gt_R.T,   label="GT right wrist",   color="tab:green", linestyle="--")

ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_title("Predicted vs Ground-Truth Wrist Trajectories")
ax.legend()
ax.set_box_aspect([1, 1, 1])       # equal axis scaling

fig.savefig("wrist_traj_pred_vs_gt.png", dpi=300, bbox_inches="tight")
plt.close(fig)
