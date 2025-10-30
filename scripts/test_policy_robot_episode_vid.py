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
from openpi_client import image_tools

# ─────────────—— file locations & basic constants —──────────────
stride = 3
demo = 3
IMG_SIZE = 224
PROMPT   = "vertical_pick_place"

# IMG_FMT = f"/iris/projects/humanoid/dataset/New_QUEST_DATA_ROBOT/demo_{demo}/left/{{:06d}}.jpg"
# LEFT_WRIST_IMG_FMT = f"/iris/projects/humanoid/dataset/New_QUEST_DATA_ROBOT/demo_{demo}/left_wrist/{{:06d}}.jpg"
# RIGHT_WRIST_IMG_FMT = f"/iris/projects/humanoid/dataset/New_QUEST_DATA_ROBOT/demo_{demo}/right_wrist/{{:06d}}.jpg"
# CSV_PATH = f"/iris/projects/humanoid/dataset/New_QUEST_DATA_ROBOT/demo_{demo}/ee_hand.csv"

IMG_FMT = f"/iris/projects/humanoid/dataset/DEMO_PICK_PLACE/yellow_duck/demo_{demo}/left/{{:06d}}.jpg"
LEFT_WRIST_IMG_FMT = f"/iris/projects/humanoid/dataset/DEMO_PICK_PLACE/yellow_duck/demo_{demo}/left_wrist/{{:06d}}.jpg"
RIGHT_WRIST_IMG_FMT = f"/iris/projects/humanoid/dataset/DEMO_PICK_PLACE/yellow_duck/demo_{demo}/right_wrist/{{:06d}}.jpg"
CSV_PATH = f"/iris/projects/humanoid/dataset/DEMO_PICK_PLACE/yellow_duck/demo_{demo}/ee_hand.csv"

# IMG_FMT = f"/iris/projects/humanoid/openpi/scripts/demo_test2/left/{{:06d}}.jpg"
# LEFT_WRIST_IMG_FMT = f"/iris/projects/humanoid/openpi/scripts/demo_test2/left_wrist/{{:06d}}.jpg"
# RIGHT_WRIST_IMG_FMT = f"/iris/projects/humanoid/openpi/scripts/demo_test2/right_wrist/{{:06d}}.jpg"
# CSV_PATH = f"/iris/projects/humanoid/openpi/scripts/demo_test2/ee_hand.csv"

right_hand_cols = [f"right_hand_{i}" for i in range(20)]
left_hand_cols = [f"left_hand_{i}" for i in range(20)]

# Intrinsics (use as-is, no scaling)
K_LEFT  = np.array([[730.2571411132812, 0.0, 637.2598876953125],
                    [0.0, 730.2571411132812, 346.41082763671875],
                    [0.0, 0.0, 1.0]], dtype=np.float64)

# Extrinsics (camera frame to the robot frame (inverse of the camera calib extrinsics))
# T_BASE_TO_CAM_LEFT = np.linalg.inv(np.array([
#     [ 0.00692993, -0.87310148,  0.48748926,  0.14062141],
#     [-0.99995006, -0.00956093, -0.00290894,  0.03612369],
#     [ 0.00720065, -0.48744476, -0.87312414,  0.46063114],
#     [ 0., 0., 0., 1. ]
# ], dtype=np.float64))

# calib for pick and place 10/15/2025
T_BASE_TO_CAM_LEFT = np.linalg.inv(np.array([
    [0.01988061, -0.43758429,  0.89895759,  0.14056752],
    [-0.9996933,   0.00457983,  0.02433772,  0.02539622],
    [-0.01476688, -0.89916573, -0.43735903,  0.43713101],
    [0.0, 0.0, 0.0, 1.0]
]))


# ─────────────────────────── helpers ────────────────────────────

def scale_K(K, new_W, new_H, orig_W=1280, orig_H=720):
    K = K.copy()
    sx, sy = new_W / orig_W, new_H / orig_H
    K[0,0] *= sx;  K[1,1] *= sy
    K[0,2] *= sx;  K[1,2] *= sy
    return K

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
    p_world_l = np.array([row["left_pos_x"], row["left_pos_y"], row["left_pos_z"]], dtype=np.float64)
    p_cam_l = _world_to_cam3(p_world_l)

    p_world_r = np.array([row["right_pos_x"], row["right_pos_y"], row["right_pos_z"]], dtype=np.float64)
    p_cam_r = _world_to_cam3(p_world_r)

    # Right wrist orientation (quat -> 6D) in left camera frame
    lq = [row["left_ori_x"], row["left_ori_y"], row["left_ori_z"], row["left_ori_w"]]
    lR = R.from_quat(lq).as_matrix()
    R_cam_l = _rot_base_to_cam(lR)
    ori6d_l = R_cam_l[:, :2].reshape(-1, order="F")
    
    rq = [row["right_ori_x"], row["right_ori_y"], row["right_ori_z"], row["right_ori_w"]]
    rR = R.from_quat(rq).as_matrix()
    R_cam = _rot_base_to_cam(rR)
    ori6d_r = R_cam[:, :2].reshape(-1, order="F")

    # Right hand command joints (20)
    joints20_l = np.asarray([row[c] for c in left_hand_cols], dtype=np.float32)
    joints20_r = np.asarray([row[c] for c in right_hand_cols], dtype=np.float32)

    # Final action vector: [pos_cam(3), ori6d(6), joints20(20)] = 29 dims
    a = np.concatenate([p_cam_l.astype(np.float32), ori6d_l.astype(np.float32), 
                        p_cam_r.astype(np.float32), ori6d_r.astype(np.float32), 
                        joints20_l, joints20_r], axis=0)
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
conf      = cfg.get_config("pi05_galaxea_egodex_abs_joints")
ckpt_dir  = download.maybe_download("/iris/projects/humanoid/openpi/checkpoints/pi05_galaxea_egodex_abs_joints/galaxea_egodex_abs_joints/3000")
policy    = policy_config.create_trained_policy(conf, ckpt_dir)

# ───────────────────────── episode loop ─────────────────────────
OUT_DIR = f"vis_demo_{demo}"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
N  = len(df)

for t in range(N):
    # ----- build per-frame inputs aligned to time t -----
    img_path_t = IMG_FMT.format(t)
    lw_img_path_t = LEFT_WRIST_IMG_FMT.format(t)
    rw_img_path_t = RIGHT_WRIST_IMG_FMT.format(t)

    # image for inference (224×224 like training)
    img_infer = cv2.imread(img_path_t)
    img_infer = cv2.cvtColor(img_infer, cv2.COLOR_BGR2RGB)
    img_infer = cv2.resize(img_infer, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img_infer = img_infer.astype(np.float32) / 255.0

    lw_img_infer = cv2.imread(lw_img_path_t)
    lw_img_infer = cv2.cvtColor(lw_img_infer, cv2.COLOR_BGR2RGB)
    lw_img_infer = cv2.resize(lw_img_infer, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    lw_img_infer = lw_img_infer.astype(np.float32) / 255.0

    rw_img_infer = cv2.imread(rw_img_path_t)
    rw_img_infer = cv2.cvtColor(rw_img_infer, cv2.COLOR_BGR2RGB)
    rw_img_infer = cv2.resize(rw_img_infer, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    rw_img_infer = rw_img_infer.astype(np.float32) / 255.0

    def _read_joint_array(row, cols):
        vals = []
        for c in cols:
            vals.append(row[c] if c in row and np.isfinite(row[c]) else 0.0)
        return np.asarray(vals, dtype=np.float32)

    # compute state
    def _cmd7(row, side: str):
        cols = ([f"left_hand_{i}"  for i in range(1, 8)]
                if side == "left" else
                [f"right_hand_{i}" for i in range(1, 8)])
        return _read_joint_array(row, cols)  # (7,)

    def _world_to_cam3(p_world3):
        """Transform a single 3D point from robot base -> left camera frame."""
        ph = np.array([p_world3[0], p_world3[1], p_world3[2], 1.0], dtype=np.float64)
        pc = T_BASE_TO_CAM_LEFT @ ph
        return pc[:3]  # (Xc, Yc, Zc)

    def _rot_base_to_cam(R_base):
        return (T_BASE_TO_CAM_LEFT[:3,:3] @ R_base).astype(np.float32)


    def _row_wrist_pose6d_in_left_cam(row, side: str):
        if side == "left":
            p_world = np.array([row["left_pos_x"], row["left_pos_y"], row["left_pos_z"]], dtype=np.float64)
            rq = [row["left_ori_x"], row["left_ori_y"], row["left_ori_z"], row["left_ori_w"]]
        else:
            p_world = np.array([row["right_pos_x"], row["right_pos_y"], row["right_pos_z"]], dtype=np.float64)
            rq = [row["right_ori_x"], row["right_ori_y"], row["right_ori_z"], row["right_ori_w"]]

        p_cam = _world_to_cam3(p_world)
        rR    = R.from_quat(rq).as_matrix()
        R_cam = _rot_base_to_cam(rR)
        ori6d = R_cam[:, :2].reshape(-1, order="F")  # 6D

        return p_cam.astype(np.float32), ori6d.astype(np.float32)  # (3,), (6,)

    row0 = df.iloc[t]
    
    cmd7L = _cmd7(row0, "left")
    cmd7R = _cmd7(row0, "right")

    pL0, oL0 = _row_wrist_pose6d_in_left_cam(row0, "left")
    pR0, oR0 = _row_wrist_pose6d_in_left_cam(row0, "right")

    state = np.concatenate([pL0, oL0, pR0, oR0, cmd7L, cmd7R], axis=0).astype(np.float32)  # (32,)

    example = {
        "state": jnp.asarray(state),
        "wrist_image_left":  image_tools.convert_to_uint8(lw_img_infer),
        "wrist_image_right": image_tools.convert_to_uint8(rw_img_infer),
        "image": jnp.asarray(img_infer),
        "prompt": PROMPT,
    }

    # ----- inference -----
    pred = np.asarray(policy.infer(example)["actions"])   # e.g., (T_pred, 26)
    traj_pred_L = pred[0::2, 0:3]                         # left-wrist cam-frame XYZt
    traj_pred_R = pred[1::2, 0:3]                         # right-wrist cam-frame XYZ

    # ----- build GT window starting at t, matching pred length/stride -----
    T_half = int(len(pred) // 2)
    gt_rows = df.iloc[t : t + stride * T_half : stride]
    if len(gt_rows) == 0:
        continue
    gt = np.stack([_row_to_action(r) for _, r in gt_rows.iterrows()], axis=0)  # (L, 29)
    traj_gt_L = gt[:, 0:3]
    traj_gt_R = gt[:, 9:12]

    # trim to common length
        # trim to common length
    L = min(len(traj_pred_R), len(traj_gt_R), len(traj_pred_L), len(traj_gt_L))
    traj_pred_L = traj_pred_L[:L]
    traj_pred_R = traj_pred_R[:L]
    traj_gt_L   = traj_gt_L[:L]
    traj_gt_R   = traj_gt_R[:L]

    # anchor predictions by the first GT point (cam-frame deltas → absolute)
    traj_pred_R[:, :3] += traj_gt_R[0, :3]
    traj_pred_L[:, :3] += traj_gt_L[0, :3]

    # ----- overlay on original-resolution image (use original K_LEFT) -----
    img_draw = cv2.imread(img_path_t)  # BGR original size
    if img_draw is None:
        print(f"[skip overlay] cannot read {img_path_t}")
        continue
    h, w = img_draw.shape[:2]
    K_use = scale_K(K_LEFT, w, h, orig_W=1280, orig_H=720)

    # project both wrists
    px_pred_R = _project_cam_to_px(traj_pred_R, K_use)
    px_gt_R   = _project_cam_to_px(traj_gt_R,   K_use)
    px_pred_L = _project_cam_to_px(traj_pred_L, K_use)
    px_gt_L   = _project_cam_to_px(traj_gt_L,   K_use)

    # clip to image bounds
    m_pred_R = _in_bounds(px_pred_R, w, h); px_pred_R = px_pred_R[m_pred_R].astype(np.int32)
    m_gt_R   = _in_bounds(px_gt_R,   w, h); px_gt_R   = px_gt_R[m_gt_R].astype(np.int32)
    m_pred_L = _in_bounds(px_pred_L, w, h); px_pred_L = px_pred_L[m_pred_L].astype(np.int32)
    m_gt_L   = _in_bounds(px_gt_L,   w, h); px_gt_L   = px_gt_L[m_gt_L].astype(np.int32)

    # colors
    COL_PRED_R = (0,   0, 255)   # red
    COL_GT_R   = (0, 255,   0)   # green
    COL_PRED_L = (255,  0, 255)  # magenta
    COL_GT_L   = (255,255,   0)  # cyan-ish/yellow

    # draw paths
    _draw_path(img_draw, px_gt_R,   COL_GT_R,   thickness=2, radius=2)
    _draw_path(img_draw, px_pred_R, COL_PRED_R, thickness=2, radius=2)
    _draw_path(img_draw, px_gt_L,   COL_GT_L,   thickness=2, radius=2)
    _draw_path(img_draw, px_pred_L, COL_PRED_L, thickness=2, radius=2)

    # legend
    legend_w, legend_h = 300, 120
    cv2.rectangle(img_draw, (10, 10), (10 + legend_w, 10 + legend_h), (255, 255, 255), -1)
    cv2.putText(img_draw, "GT right wrist",   (20, 35),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_GT_R,   2, cv2.LINE_AA)
    cv2.putText(img_draw, "Pred right wrist", (20, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_PRED_R, 2, cv2.LINE_AA)
    cv2.putText(img_draw, "GT left wrist",    (20, 85),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_GT_L,   2, cv2.LINE_AA)
    cv2.putText(img_draw, "Pred left wrist",  (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_PRED_L, 2, cv2.LINE_AA)

    # save per-frame
    out_path = os.path.join(OUT_DIR, f"{t:06d}.jpg")
    cv2.imwrite(out_path, img_draw)

print(f"Saved per-frame full trajectories to: {OUT_DIR}")

# (optional) quick ffmpeg stitch (run in shell):
# ffmpeg -y -framerate 30 -i vis_demo_{demo}/%06d.jpg -c:v libx264 -pix_fmt yuv420p vis_demo_{demo}.mp4
