"""
usage:
uv run -active src/openpi/training/galaxea_dataset.py

There are two blocks of code at the very bottom, first block is for overlaid keypoint and edge visualization
the bottom one is for the action visualization project on the image. Uncomment accordingly to your needs.

uv run src/openpi/training/galaxea_dataset.py   --dataset_dir /iris/projects/humanoid/dataset/DEMO_PICK_PLACE/banana   --out_dir galaxea_action_state_vis  --num_samples 40   --chunk_size 5   --stride 3
"""
import os
import cv2
import numpy as np
import torch
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
if not hasattr(np, "float"):
    np.float = float

from urdfpy import URDF
from openpi.training.hand_keypoints_config import JOINT_COLOR_BGR
import random

def scale_K(K, new_W, new_H, orig_W=1280, orig_H=720):
    scale_x = new_W / orig_W
    scale_y = new_H / orig_H
    K_scaled = K.copy()
    K_scaled[0,0] *= scale_x   # fx
    K_scaled[1,1] *= scale_y   # fy
    K_scaled[0,2] *= scale_x   # cx
    K_scaled[1,2] *= scale_y   # cy
    return K_scaled

# Robot link -> MANO right-hand semantic name (used only for colors)
# Map robot link -> MANO semantic name used by JOINT_COLOR_BGR (both hands)
ROBOT_TO_MANO_R = {
    "rl_dg_base": "rightHand",
    "rl_thumb_knuckle1": "rightThumbKnuckle",
    "rl_thumb_knuckle2": "rightThumbIntermediateBase",
    "rl_thumb_tip":      "rightThumbTip",
    "rl_index_knuckle1": "rightIndexFingerKnuckle",
    "rl_index_knuckle2": "rightIndexFingerIntermediateBase",
    "rl_index_tip":      "rightIndexFingerTip",
    "rl_middle_knuckle1": "rightMiddleFingerKnuckle",
    "rl_middle_knuckle2": "rightMiddleFingerIntermediateBase",
    "rl_middle_tip":      "rightMiddleFingerTip",
    "rl_ring_knuckle1": "rightRingFingerKnuckle",
    "rl_ring_knuckle2": "rightRingFingerIntermediateBase",
    "rl_ring_tip":      "rightRingFingerTip",
    "rl_pinky_knuckle1": "rightLittleFingerKnuckle",
    "rl_pinky_knuckle2": "rightLittleFingerIntermediateBase",
    "rl_pinky_tip":      "rightLittleFingerTip",
    # DG5F link-style names (the ones you actually use below)
    "rl_dg_1_2": "rightThumbKnuckle",
    "rl_dg_1_3": "rightThumbIntermediateBase",
    "rl_dg_1_4": "rightThumbIntermediateTip",
    "rl_dg_1_tip": "rightThumbTip",
    "rl_dg_2_2": "rightIndexFingerKnuckle",
    "rl_dg_2_3": "rightIndexFingerIntermediateBase",
    "rl_dg_2_4": "rightIndexFingerIntermediateTip",
    "rl_dg_2_tip": "rightIndexFingerTip",
    "rl_dg_3_2": "rightMiddleFingerKnuckle",
    "rl_dg_3_3": "rightMiddleFingerIntermediateBase",
    "rl_dg_3_4": "rightMiddleFingerIntermediateTip",
    "rl_dg_3_tip": "rightMiddleFingerTip",
    "rl_dg_4_2": "rightRingFingerKnuckle",
    "rl_dg_4_3": "rightRingFingerIntermediateBase",
    "rl_dg_4_4": "rightRingFingerIntermediateTip",
    "rl_dg_4_tip": "rightRingFingerTip",
    "rl_dg_5_2": "rightLittleFingerKnuckle",
    "rl_dg_5_3": "rightLittleFingerIntermediateBase",
    "rl_dg_5_4": "rightLittleFingerIntermediateTip",
    "rl_dg_5_tip": "rightLittleFingerTip",
}

ROBOT_TO_MANO_L = {
    "ll_dg_base": "leftHand",
    "ll_dg_1_2": "leftThumbKnuckle",
    "ll_dg_1_3": "leftThumbIntermediateBase",
    "ll_dg_1_4": "leftThumbIntermediateTip",
    "ll_dg_1_tip": "leftThumbTip",
    "ll_dg_2_2": "leftIndexFingerKnuckle",
    "ll_dg_2_3": "leftIndexFingerIntermediateBase",
    "ll_dg_2_4": "leftIndexFingerIntermediateTip",
    "ll_dg_2_tip": "leftIndexFingerTip",
    "ll_dg_3_2": "leftMiddleFingerKnuckle",
    "ll_dg_3_3": "leftMiddleFingerIntermediateBase",
    "ll_dg_3_4": "leftMiddleFingerIntermediateTip",
    "ll_dg_3_tip": "leftMiddleFingerTip",
    "ll_dg_4_2": "leftRingFingerKnuckle",
    "ll_dg_4_3": "leftRingFingerIntermediateBase",
    "ll_dg_4_4": "leftRingFingerIntermediateTip",
    "ll_dg_4_tip": "leftRingFingerTip",
    "ll_dg_5_2": "leftLittleFingerKnuckle",
    "ll_dg_5_3": "leftLittleFingerIntermediateBase",
    "ll_dg_5_4": "leftLittleFingerIntermediateTip",
    "ll_dg_5_tip": "leftLittleFingerTip",
}

ROBOT_TO_MANO = {**ROBOT_TO_MANO_R, **ROBOT_TO_MANO_L}

WRIST = "rl_dg_base"  # wrist (plot); ignore rl_dg_mount
FINGERS_ROBOT = {
    "Thumb":       ["rl_dg_1_2","rl_dg_1_3","rl_dg_1_4","rl_dg_1_tip"],
    "IndexFinger": ["rl_dg_2_2","rl_dg_2_3","rl_dg_2_4","rl_dg_2_tip"],
    "MiddleFinger":["rl_dg_3_2","rl_dg_3_3","rl_dg_3_4","rl_dg_3_tip"],
    "RingFinger":  ["rl_dg_4_2","rl_dg_4_3","rl_dg_4_4","rl_dg_4_tip"],
    "LittleFinger":["rl_dg_5_2","rl_dg_5_3","rl_dg_5_4","rl_dg_5_tip"],
}

# --- Left & Right hand equivalents for overlay ---
WRIST_R = "rl_dg_base"
WRIST_L = "ll_dg_base"

FINGERS_ROBOT_R = FINGERS_ROBOT  # reuse right-hand version

FINGERS_ROBOT_L = {
    "Thumb":       ["ll_dg_1_2","ll_dg_1_3","ll_dg_1_4","ll_dg_1_tip"],
    "IndexFinger": ["ll_dg_2_2","ll_dg_2_3","ll_dg_2_4","ll_dg_2_tip"],
    "MiddleFinger":["ll_dg_3_2","ll_dg_3_3","ll_dg_3_4","ll_dg_3_tip"],
    "RingFinger":  ["ll_dg_4_2","ll_dg_4_3","ll_dg_4_4","ll_dg_4_tip"],
    "LittleFinger":["ll_dg_5_2","ll_dg_5_3","ll_dg_5_4","ll_dg_5_tip"],
}

# Map robot link -> MANO right-hand semantic name used by JOINT_COLOR_BGR
ROBOT_TO_MANO_MIN = {WRIST: "rightHand"}
for mano_finger, chain in FINGERS_ROBOT.items():
    segs = ["Knuckle","IntermediateBase","IntermediateTip","Tip"]
    for seg, link in zip(segs, chain):
        ROBOT_TO_MANO_MIN[link] = f"right{mano_finger}{seg}"

def _link_to_mano(name: str) -> str | None:
    # 1) explicit table first
    if name in ROBOT_TO_MANO_MIN:
        return ROBOT_TO_MANO_MIN[name]

    # 2) heuristic for DG-style names: rl_* or ll_*
    side = None
    base = name
    if name.startswith("rl_"):
        side = "right"; base = name[3:]   # e.g. "dg_2_tip" or "index_tip"
    elif name.startswith("ll_"):
        side = "left";  base = name[3:]

    if side is None:
        return None

    parts = base.split("_")  # try to parse things like "dg_2_tip" or "index_knuckle2"
    if len(parts) < 2:
        return None

    # support both "index_tip" layout and "dg_2_tip" layout
    # finger id/name -> MANO finger word
    finger_map_num = {"1":"Thumb","2":"IndexFinger","3":"MiddleFinger","4":"RingFinger","5":"LittleFinger"}
    finger_map_txt = {
        "thumb":"Thumb","index":"IndexFinger","middle":"MiddleFinger","ring":"RingFinger",
        "pinky":"LittleFinger","little":"LittleFinger"
    }

    # determine finger
    if parts[0] == "dg" and len(parts) >= 3:
        finger_word = finger_map_num.get(parts[1])
        part = parts[2]
    else:
        finger_word = finger_map_txt.get(parts[0])
        part = "_".join(parts[1:])

    if finger_word is None:
        return None

    # map segment keywords to MANO segment
    part_map = {
        "knuckle1": "Knuckle",
        "knuckle2": "IntermediateBase",
        "knuckle3": "IntermediateTip",
        "2":       "Knuckle",            # dg_*_2
        "3":       "IntermediateBase",   # dg_*_3
        "4":       "IntermediateTip",    # dg_*_4
        "tip":     "Tip",
    }
    seg = part_map.get(part)
    if seg is None:
        return None

    side_prefix = "left" if side == "left" else "right"
    return f"{side_prefix}{finger_word}{seg}"


# TODO: add keypoint offsets as needed (also fix visualization)
# --- OPTIONAL: per-hand visualization offsets (hand frame, meters) ---
VIS_OFFSET_L = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # tweak me
VIS_OFFSET_R = np.array([0.00, 0.00, 0.00], dtype=np.float64)  # tweak me

def _apply_vis_offset(T_hand: np.ndarray, offset_xyz: np.ndarray) -> np.ndarray:
    """Translate the whole hand by 'offset_xyz' given in the hand frame."""
    if offset_xyz is None or not np.any(np.isfinite(offset_xyz)) or not np.any(offset_xyz):
        return T_hand
    T_out = T_hand.copy()
    T_out[:3, 3] += T_out[:3, :3] @ offset_xyz  # rotate offset into world, then add
    return T_out

def draw_skeleton_occlusion_aware(
    image_rgb_float: np.ndarray,
    names: list[str],
    uv: list[tuple[int,int] | None],
    z: np.ndarray,
    edges_by_name: list[tuple[str,str]],
    color_of: dict[str, tuple[int,int,int]],
    *,
    pt_radius: int = 6,
    line_thickness: int = 3,
    edge_segments: int = 12,
) -> np.ndarray:
    """Global depth-sorted drawing of edges + points."""
    H, W = image_rgb_float.shape[:2]
    name_to_idx = {n:i for i,n in enumerate(names)}

    # ---- DEBUG: high-level inventory ----
    n_pts_total   = len(names)
    n_pts_valid   = sum(p is not None for p in uv)
    n_edges_total = len(edges_by_name)
    n_edges_named = sum((a in name_to_idx and b in name_to_idx) for a,b in edges_by_name)
    # print(f"[DRAW] points: total={n_pts_total}, valid_uv={n_pts_valid} | "
    #       f"edges: total={n_edges_total}, with_valid_names={n_edges_named}")


    prims = []
    for a,b in edges_by_name:
        if a not in name_to_idx or b not in name_to_idx: 
            continue
        ia, ib = name_to_idx[a], name_to_idx[b]
        pa, pb = uv[ia], uv[ib]
        if (pa is None) or (pb is None): 
            continue
        ua,va = pa; ub,vb = pb
        za,zb = float(z[ia]), float(z[ib])
        for k in range(edge_segments):
            t0 = k/edge_segments; t1=(k+1)/edge_segments
            u0=int(round(ua*(1-t0)+ub*t0)); v0=int(round(va*(1-t0)+vb*t0))
            u1=int(round(ua*(1-t1)+ub*t1)); v1=int(round(va*(1-t1)+vb*t1))
            if not (0<=u0<W and 0<=v0<H and 0<=u1<W and 0<=v1<H): 
                continue
            zmid = za*(1-(t0+t1)/2)+zb*((t0+t1)/2)
            col=color_of.get(b,(210,210,210))
            prims.append(("edge",zmid,(u0,v0),(u1,v1),col))
    for i,p in enumerate(uv):
        if p is None: continue
        u,v=p
        if 0<=u<W and 0<=v<H:
            prims.append(("pt",float(z[i]),(u,v),color_of.get(names[i],(210,210,210))))
    prims.sort(key=lambda x:-x[1])
    img_bgr=cv2.cvtColor((image_rgb_float*255).astype(np.uint8),cv2.COLOR_RGB2BGR)
    for prim in prims:
        if prim[0]=="edge":
            _,_,p0,p1,col=prim
            cv2.line(img_bgr,p0,p1,col,line_thickness,cv2.LINE_AA)
        else:
            _,_,(u,v),col=prim
            cv2.circle(img_bgr,(u,v),pt_radius,col,-1,cv2.LINE_AA)
    return cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0


# --- tiny helpers / constants for FK overlay ---
def _pose_to_T(pos_xyz, quat_xyzw):
    T = np.eye(4)
    T[:3,:3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = np.asarray(pos_xyz, dtype=np.float64)
    return T

# End-effector → hand (same as your script; tweak if needed)
_theta_y = np.pi
_theta_z = -np.pi/2
_right_theta_z = np.pi
_R_y = np.array([[ np.cos(_theta_y), 0, np.sin(_theta_y)],
                 [ 0, 1, 0],
                 [-np.sin(_theta_y), 0, np.cos(_theta_y)]])
_R_z = np.array([[np.cos(_theta_z),-np.sin(_theta_z),0],
                 [np.sin(_theta_z), np.cos(_theta_z),0],
                 [0,0,1]])
_R_right_z = np.array([[np.cos(_right_theta_z),-np.sin(_right_theta_z),0],
                       [np.sin(_right_theta_z), np.cos(_right_theta_z),0],
                       [0,0,1]])
T_EE_TO_HAND_L = np.eye(4); T_EE_TO_HAND_L[:3,:3] = _R_y @ _R_z;       T_EE_TO_HAND_L[:3,3] = [ 0.00,-0.033, 0.00]
T_EE_TO_HAND_R = np.eye(4); T_EE_TO_HAND_R[:3,:3] = _R_y @ _R_z @ _R_right_z; T_EE_TO_HAND_R[:3,3] = [-0.02, 0.02, 0.025]

# Simple link groups to draw (right hand); prefixes in URDF are "rl_*"
_HAND_LINK_GROUPS = {
    "tips":     ["thumb_tip","index_tip","middle_tip","ring_tip","pinky_tip"],
    "mids":     ["thumb_knuckle2","index_knuckle2","middle_knuckle2","ring_knuckle2","pinky_knuckle2"],
    "knuckles": ["thumb_knuckle1","index_knuckle1","middle_knuckle1","ring_knuckle1","pinky_knuckle1"],
}
def _connect(seq):  # lines between consecutive links
    return [(seq[i], seq[i+1]) for i in range(len(seq)-1)]

# Intrinsics
K_LEFT  = np.array([[730.2571411132812, 0.0, 637.2598876953125],
                    [0.0, 730.2571411132812, 346.41082763671875],
                    [0.0, 0.0, 1.0]], dtype=np.float64)

K_RIGHT = np.array([[730.257, 0.0, 637.259],
                    [0.0, 730.257, 346.410],
                    [0.0, 0.0, 1.0]], dtype=np.float64)

# Extrinsics (camera frame to the robot frame (inverse of the camera calib extrinsics))
# block transfer data (from wayyy back)
# T_BASE_TO_CAM_LEFT = np.linalg.inv(np.array([
#     [ 0.00692993, -0.87310148,  0.48748926,  0.14062141],
#     [-0.99995006, -0.00956093, -0.00290894,  0.03612369],
#     [ 0.00720065, -0.48744476, -0.87312414,  0.46063114],
#     [ 0., 0., 0., 1. ]
# ], dtype=np.float64))


# calib for left camera quest assembly data 10/15/2025
# T_BASE_TO_CAM_LEFT = np.linalg.inv(np.array([
#     [0.02169645, -0.70143451,  0.71240361,  0.14534308],
#     [-0.99949077,  0.00145864,  0.03187594,  0.03543434],
#     [-0.02339802, -0.71273242, -0.70104567,  0.47400349],
#     [0.0,          0.0,          0.0,          1.0]
# ]))

# calib for pick and place 10/15/2025
T_BASE_TO_CAM_LEFT = np.linalg.inv(np.array([
    [0.01988061, -0.43758429,  0.89895759,  0.14056752],
    [-0.9996933,   0.00457983,  0.02433772,  0.02539622],
    [-0.01476688, -0.89916573, -0.43735903,  0.43713101],
    [0.0, 0.0, 0.0, 1.0]
]))


# calib for left camera quest assembly data
# T_BASE_TO_CAM_LEFT = np.linalg.inv(np.array([
#     [0.02169645, -0.70143451,  0.71240361,  0.14534308],
#     [-0.99949077,  0.00145864,  0.03187594,  0.03543434],
#     [-0.02339802, -0.71273242, -0.70104567,  0.47400349],
#     [0.0,          0.0,          0.0,          1.0]], dtype=np.float64))

# similar as above for the right camera
T_BASE_TO_CAM_RIGHT = np.linalg.inv(np.array([
    [-0.00334115, -0.8768872 ,  0.48068458,  0.14700305],
    [-0.99996141,  0.0068351 ,  0.00551836, -0.02680847],
    [-0.00812451, -0.4806476 , -0.87687621,  0.46483729],
    [ 0., 0., 0., 1. ]
], dtype=np.float64))

# NEW: common fingertip link-name helper
def _tip_links(prefix: str) -> list[str]:
    # consistent with your right-hand naming (rl_dg_?_tip)
    return [f"{prefix}_dg_{i}_tip" for i in [1, 2, 3, 4, 5]]  # thumb..pinky

TIP_LINKS_L = _tip_links("ll")
TIP_LINKS_R = _tip_links("rl")

class GalaxeaDatasetKeypointsJoints(torch.utils.data.Dataset):
    def __init__(self, task: str, dataset_dir, chunk_size, stride=3, overlay=False,
                #  hand_mode: str = "right",  # "left", "right", or "both"
                 urdf_left_path="/iris/projects/humanoid/act/dg_description/urdf/dg5f_left.urdf",
                 urdf_right_path="/iris/projects/humanoid/act/dg_description/urdf/dg5f_right.urdf"):
        super(GalaxeaDatasetKeypointsJoints).__init__()
        self.dataset_dir = dataset_dir
        self.chunk_size = chunk_size
        self.img_height = 224
        self.img_width  = 224
        self.stride = stride
        self.overlay = overlay
        if not isinstance(task, str) or task.strip() == "":
            raise ValueError("`task` must be a non-empty string.")
        self.task = task
        # self.hand_mode = hand_mode  # NEW

        # columns
        self.left_hand_cols  = [f"left_hand_{i}"  for i in range(20)] # commanded joint positions
        self.right_hand_cols = [f"right_hand_{i}" for i in range(20)]
        self.left_actual_hand_cols  = [f"left_actual_hand_{i}"  for i in range(20)] # true state of the hand
        self.right_actual_hand_cols = [f"right_actual_hand_{i}" for i in range(20)]

        self.action_camera = "left"

        # # CHANGED: load both URDFs when needed
        # if self.overlay or (self.hand_mode in ("left", "both")):
        self.robot_l = URDF.load(urdf_left_path)
        self.left_joint_names = [j.name for j in self.robot_l.joints if j.joint_type != "fixed"]
        # if self.overlay or (self.hand_mode in ("right", "both")):
        self.robot_r = URDF.load(urdf_right_path)
        self.right_joint_names = [j.name for j in self.robot_r.joints if j.joint_type != "fixed"]

        # right-hand drawing connections (keep your original overlay code)
        if self.overlay:
            self.right_connections = [(j.parent, j.child) for j in self.robot_r.joints]
            for g in _HAND_LINK_GROUPS.values():
                names = [f"rl_{x}" for x in g]
                self.right_connections.extend(_connect(names))

        self.debug_labels = True

        # collect demo folders: Demo1, Demo2, ..., DemoN
        def _demo_key(name: str):
            # handles "demo_12" or "Demo12"
            try:
                if name.startswith("demo_"):
                    return int(name.split("demo_")[1])
                if name.startswith("Demo"):
                    return int(name.replace("Demo", ""))
            except Exception:
                pass
            return float("inf")

        self.episode_dirs = sorted(
            [
                os.path.join(self.dataset_dir, d)
                for d in os.listdir(self.dataset_dir)
                if os.path.isdir(os.path.join(self.dataset_dir, d))
                and (d.startswith("demo_") or d.startswith("Demo"))
            ],
            key=lambda p: _demo_key(os.path.basename(p)),
        )

        # TODO: delete me
        # self.episode_dirs = random.sample(self.episode_dirs, 25)
        # print("Found episode dirs:", self.episode_dirs, len(self.episode_dirs))

        # Precompute (episode_len) for each episode by reading its CSV once
        self.episodes = []  # list of (demo_dir, episode_len)
        for demo_dir in self.episode_dirs:
            csv_path = os.path.join(demo_dir, "ee_hand.csv")
            if not os.path.exists(csv_path):
                continue
            try:
                # fast-enough simple read; if huge, you can switch to a faster row count strategy later
                episode_len = len(pd.read_csv(csv_path))
            except Exception:
                continue
            if episode_len <= 0:
                continue
            self.episodes.append((demo_dir, episode_len))

        if not self.episodes:
            raise RuntimeError(f"No valid episodes under {self.dataset_dir}")

        # print(self.episodes)

        # Build flat index of (ep_id, t0) for all valid starting timesteps
        # We require full chunks (no padding). If you want padding, see the variant below.
        horizon = self.chunk_size * self.stride
        self.index = []
        for ep_id, (_, N) in enumerate(self.episodes):
            # print("going thru:", ep_id)
            # TODO: fix this
            last_start = N - 1
            if last_start < 0:
                continue
            # t0 = 0, stride, 2*stride, ..., <= last_start
            self.index.extend((ep_id, t0) for t0 in range(0, last_start + 1, self.stride))

    def __len__(self):
        return len(self.index)

    def _read_joint_array(self, row, cols):
        vals = []
        for c in cols:
            vals.append(row[c] if c in row and np.isfinite(row[c]) else 0.0)
        return np.asarray(vals, dtype=np.float32)

    def _joint_cols20(self, side: str, kind: str) -> list[str]:
        """
        kind: 'desired' -> commanded joint columns
            'current' -> actual/current joint state columns
        """
        if side == "left":
            # return (self.left_actual_hand_cols if kind == "desired" else self.left_hand_cols)
            return (self.left_hand_cols if kind == "desired" else self.left_actual_hand_cols)
        else:
            # return (self.right_actual_hand_cols if kind == "desired" else self.right_hand_cols)
            return (self.right_hand_cols if kind == "desired" else self.right_actual_hand_cols)

    def _joints20_from_row(self, row, side: str, kind: str) -> np.ndarray:
        return self._read_joint_array(row, self._joint_cols20(side, kind))


    def _fk_points_world_and_wristT(self, row, side: str, kind: str = "actual"):
        """Return:
            pts_world: {link_name -> (3,)} world positions for links of `side`
            T_wrist_world: (4,4) wrist pose in world (rotation+translation)
        """
        # EE -> HAND
        if side == "left":
            T_ee = _pose_to_T([row["left_pos_x"], row["left_pos_y"], row["left_pos_z"]],
                            [row["left_ori_x"], row["left_ori_y"], row["left_ori_z"], row["left_ori_w"]])
            T_hand = T_ee @ T_EE_TO_HAND_L
            robot, joint_names, prefix, wrist_name = self.robot_l, self.left_joint_names, "ll_", WRIST_L
            T_hand = _apply_vis_offset(T_hand, VIS_OFFSET_L)
        else:
            T_ee = _pose_to_T([row["right_pos_x"], row["right_pos_y"], row["right_pos_z"]],
                            [row["right_ori_x"], row["right_ori_y"], row["right_ori_z"], row["right_ori_w"]])
            T_hand = T_ee @ T_EE_TO_HAND_R
            robot, joint_names, prefix, wrist_name = self.robot_r, self.right_joint_names, "rl_", WRIST_R
            T_hand = _apply_vis_offset(T_hand, VIS_OFFSET_R)

        angles = self._get_joint_angles20(row, side, kind)  # "actual" or "cmd"
        fk = robot.link_fk(cfg=dict(zip(joint_names, angles)), use_names=True)

        T_fkbase_inv = np.linalg.inv(fk.get("FK_base", np.eye(4)))

        # Collect world positions and also compute wrist world transform
        pts_world = {}
        T_wrist_world = None
        for link_name, T_link_model in fk.items():
            if not link_name.startswith(prefix):
                continue
            T_link_in_hand = T_fkbase_inv @ T_link_model
            T_world = T_hand @ T_link_in_hand
            if link_name == wrist_name:
                T_wrist_world = T_world.copy()
            xyz = T_world[:3, 3]
            if np.all(np.isfinite(xyz)):
                pts_world[link_name] = xyz

        # Fallback if wrist wasn't found for some reason
        if T_wrist_world is None:
            T_wrist_world = T_hand

        return pts_world, T_wrist_world


    def _tips_in_wrist_frame(self, pts_map_world: dict, T_wrist_world: np.ndarray, side: str) -> np.ndarray:
        """Return 15D [thumb..pinky] tip positions in wrist local frame."""
        tips = TIP_LINKS_L if side == "left" else TIP_LINKS_R
        Rw = T_wrist_world[:3, :3]
        tw = T_wrist_world[:3, 3]
        out = []
        for link in tips:
            if link not in pts_map_world:
                out.extend([np.nan, np.nan, np.nan])
            else:
                pw = pts_map_world[link]
                plocal = Rw.T @ (pw - tw)
                out.extend(plocal.astype(np.float32))
        return np.asarray(out, dtype=np.float32)  # (15,)


    def _row_to_hand_action15_local(self, row, side: str, kind: str = "cmd") -> np.ndarray:
        """
        15D fingertip positions (thumb->pinky) expressed in the wrist local frame.
        """
        pts_world, T_wrist_world = self._fk_points_world_and_wristT(row, side, kind=kind)
        return self._tips_in_wrist_frame(pts_world, T_wrist_world, side)  # (15,)

    def _row_to_hand_action24_mixed(self, row, side: str, kind: str = "cmd") -> np.ndarray:
        """
        Returns 24D:
        [ wrist_pos_in_left_cam(3), wrist_ori6d_in_left_cam(6), fingertips_local_wrt_wrist(15) ]
        kind: "cmd" or "actual" joint angles for FK.
        """
        # wrist in camera frame
        p_cam, ori6d = self._row_wrist_pose6d_in_left_cam(row, side)

        # fingertips in wrist local frame
        pts_world, T_wrist_world = self._fk_points_world_and_wristT(row, side, kind=kind)
        tip15_local = self._tips_in_wrist_frame(pts_world, T_wrist_world, side)

        return np.concatenate([p_cam, ori6d, tip15_local], axis=0).astype(np.float32)  # (24,)


    # --- helpers: pick commanded vs actual joints safely ---
    def _get_joint_cols(self, side: str, kind: str) -> list[str]:
        if side == "left":
            primary  = self.left_actual_hand_cols  if kind == "actual" else self.left_hand_cols
            fallback = self.left_hand_cols         if kind == "actual" else self.left_actual_hand_cols
        else:
            primary  = self.right_actual_hand_cols if kind == "actual" else self.right_hand_cols
            fallback = self.right_hand_cols        if kind == "actual" else self.right_actual_hand_cols
        return primary, fallback

    def _get_joint_angles20(self, row, side: str, kind: str) -> list[float]:
        primary, fallback = self._get_joint_cols(side, kind)

        def _read(cols):
            if not all(c in row for c in cols):
                return None
            vals = np.array([row[c] for c in cols], dtype=np.float64)
            return vals if np.all(np.isfinite(vals)) else None

        v = _read(primary)
        if v is None and kind == "actual":
            v = _read(fallback)  # fallback to cmd if actual missing/NaN
        if v is None:
            v = np.zeros(20, dtype=np.float64)  # last resort
        return v.astype(np.float32).tolist()

    # NEW: FK for left hand world points
    def _fk_points_left_world(self, row):
        T_ee  = _pose_to_T([row["left_pos_x"], row["left_pos_y"], row["left_pos_z"]],
                        [row["left_ori_x"], row["left_ori_y"], row["left_ori_z"], row["left_ori_w"]])
        T_hand = T_ee @ T_EE_TO_HAND_L
        angles = [float(row[f"left_actual_hand_{i}"]) for i in range(20)]
        fk = self.robot_l.link_fk(cfg=dict(zip(self.left_joint_names, angles)), use_names=True)
        T_fkbase_inv = np.linalg.inv(fk.get("FK_base", np.eye(4)))
        pts = {}
        for link_name, T_link_model in fk.items():
            if not link_name.startswith("ll_"):
                continue
            T_link_in_hand = T_fkbase_inv @ T_link_model
            T_world        = T_hand @ T_link_in_hand
            pts[link_name] = T_world[:3, 3]
        return pts

    # NEW: unified FK dispatcher
    def _fk_points_world(self, row, side: str, kind: str = "actual"):
        # EE -> HAND
        if side == "left":
            T_ee = _pose_to_T([row["left_pos_x"], row["left_pos_y"], row["left_pos_z"]],
                            [row["left_ori_x"], row["left_ori_y"], row["left_ori_z"], row["left_ori_w"]])
            T_hand = T_ee @ T_EE_TO_HAND_L
            robot, joint_names, prefix = self.robot_l, self.left_joint_names, "ll_"
            T_hand = _apply_vis_offset(T_hand, VIS_OFFSET_L)
        else:
            T_ee = _pose_to_T([row["right_pos_x"], row["right_pos_y"], row["right_pos_z"]],
                            [row["right_ori_x"], row["right_ori_y"], row["right_ori_z"], row["right_ori_w"]])
            T_hand = T_ee @ T_EE_TO_HAND_R
            T_hand = _apply_vis_offset(T_hand, VIS_OFFSET_R)
            robot, joint_names, prefix = self.robot_r, self.right_joint_names, "rl_"

        angles = self._get_joint_angles20(row, side, kind)  # kind: "actual" or "cmd"
        fk = robot.link_fk(cfg=dict(zip(joint_names, angles)), use_names=True)

        T_fkbase_inv = np.linalg.inv(fk.get("FK_base", np.eye(4)))
        pts = {}
        for link_name, T_link_model in fk.items():
            if not link_name.startswith(prefix):
                continue
            T_link_in_hand = T_fkbase_inv @ T_link_model
            T_world        = T_hand @ T_link_in_hand
            xyz = T_world[:3, 3]
            if np.all(np.isfinite(xyz)):
                pts[link_name] = xyz
        return pts

    # NEW: side-aware camera transform for wrist pose + 6D ori
    def _row_wrist_pose6d_in_left_cam(self, row, side: str):
        if side == "left":
            p_world = np.array([row["left_pos_x"], row["left_pos_y"], row["left_pos_z"]], dtype=np.float64)
            rq = [row["left_ori_x"], row["left_ori_y"], row["left_ori_z"], row["left_ori_w"]]
        else:
            p_world = np.array([row["right_pos_x"], row["right_pos_y"], row["right_pos_z"]], dtype=np.float64)
            rq = [row["right_ori_x"], row["right_ori_y"], row["right_ori_z"], row["right_ori_w"]]

        p_cam = self._world_to_cam3(p_world)
        rR    = R.from_quat(rq).as_matrix()
        R_cam = self._rot_base_to_cam(rR)
        ori6d = R_cam[:, :2].reshape(-1, order="F")  # 6D

        return p_cam.astype(np.float32), ori6d.astype(np.float32)  # (3,), (6,)

    # NEW: fingertip positions in left camera frame, order [thumb,index,middle,ring,pinky]
    def _tip_positions_in_left_cam(self, pts_map_world: dict, side: str) -> np.ndarray:
        tips = TIP_LINKS_L if side == "left" else TIP_LINKS_R
        out = []
        for link in tips:
            if link not in pts_map_world:
                out.extend([np.nan, np.nan, np.nan])  # or zeros if you prefer
            else:
                out.extend(self._world_to_cam3(pts_map_world[link]).astype(np.float32))
        return np.asarray(out, dtype=np.float32)  # (15,)

    # NEW: build 24D hand action for a given side
    # 1) Let _row_to_hand_action24 choose cmd vs actual
    # keep this from before
    def _row_to_hand_action24(self, row, side: str, kind: str = "cmd"):
        p_cam, ori6d = self._row_wrist_pose6d_in_left_cam(row, side)
        pts_map = self._fk_points_world(row, side, kind=kind)  # "cmd" or "actual"
        tip15 = self._tip_positions_in_left_cam(pts_map, side)
        return np.concatenate([p_cam, ori6d, tip15], axis=0).astype(np.float32)  # (24,)


    def _fk_points_right_world(self, row):
        # EE pose -> hand pose (world)
        T_ee  = _pose_to_T([row["right_pos_x"], row["right_pos_y"], row["right_pos_z"]],
                        [row["right_ori_x"], row["right_ori_y"], row["right_ori_z"], row["right_ori_w"]])
        T_hand = T_ee @ T_EE_TO_HAND_R
        # FK with 20 joint angles
        angles = [float(row[f"right_actual_hand_{i}"]) for i in range(20)]
        fk = self.robot_r.link_fk(cfg=dict(zip(self.right_joint_names, angles)), use_names=True)
        T_fkbase_inv = np.linalg.inv(fk.get("FK_base", np.eye(4)))
        pts = {}
        for link_name, T_link_model in fk.items():
            if not link_name.startswith("rl_"):  # right-hand links prefixed "rl_"
                continue
            T_link_in_hand = T_fkbase_inv @ T_link_model
            T_world        = T_hand @ T_link_in_hand
            pts[link_name] = T_world[:3, 3]
        return pts  # {link_name: (x,y,z)}

    def _put_label(self, img, text, xy, color=(255,255,255)):
        """Draw small text with an outline for readability."""
        x, y = int(xy[0]), int(xy[1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs, th = 0.22, 1
        # black outline
        cv2.putText(img, text, (x+1, y+1), font, fs, (0,0,0), th+2, cv2.LINE_AA)
        # main text
        cv2.putText(img, text, (x, y), font, fs, color, th, cv2.LINE_AA)


    def _project_draw_hands_on_left(self, img_bgr, pts_map_left=None, pts_map_right=None):
        """
        Projects and draws left/right/both robot hands in the left camera image,
        using occlusion-aware depth ordering (same logic as original).
        """
        h, w = img_bgr.shape[:2]
        Kvis = scale_K(K_LEFT, w, h, orig_W=1280, orig_H=720)
        fx, fy, cx, cy = Kvis[0,0], Kvis[1,1], Kvis[0,2], Kvis[1,2]

        names, edges_by_name, uv, z, color_of = [], [], [], [], {}

        # utility to process one hand
        def _process_hand(prefix, wrist, fingers, pts_map):
            local_names = [wrist] + [n for chain in fingers.values() for n in chain]
            missing = [n for n in local_names if n not in pts_map]
            # print(f"[HAND {prefix}] nodes={len(local_names)}  missing={len(missing)}  sample_missing={missing[:6]}")

            local_edges = []
            for chain in fingers.values():
                local_edges.append((wrist, chain[0]))
                local_edges.extend(zip(chain, chain[1:]))
            for name in local_names:
                if name not in pts_map:
                    uv.append(None); z.append(np.inf)
                    names.append(name)
                    continue
                x, y, z0 = pts_map[name]
                Pc = T_BASE_TO_CAM_LEFT @ np.array([x, y, z0, 1.0])
                Z = float(Pc[2])
                if Z <= 1e-6:
                    uv.append(None); z.append(np.inf); names.append(name)
                    continue
                u = int(fx * Pc[0] / Z + cx)
                v = int(fy * Pc[1] / Z + cy)
                if 0 <= u < w and 0 <= v < h:
                    uv.append((u, v)); z.append(Z)
                else:
                    uv.append(None); z.append(Z)
                names.append(name)
            edges_by_name.extend(local_edges)
            for n in local_names:
                mano = _link_to_mano(n)
                color_of[n] = JOINT_COLOR_BGR.get(mano, (210,210,210)) if mano else (210,210,210)

        # handle left/right
        if pts_map_left is not None:
            _process_hand("ll", WRIST_L, FINGERS_ROBOT_L, pts_map_left)
        if pts_map_right is not None:
            _process_hand("rl", WRIST_R, FINGERS_ROBOT_R, pts_map_right)

        # draw with your occlusion-aware routine
        z = np.array(z, dtype=np.float32)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_rgb = draw_skeleton_occlusion_aware(
            img_rgb,
            names=names,
            uv=uv,
            z=z,
            edges_by_name=edges_by_name,
            color_of=color_of,
            pt_radius=5,
            line_thickness=10,
            edge_segments=12,
        )
        return cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)



    def _load_img_raw_bgr(self, demo_dir, ts, cam_name="left"):
        p = os.path.join(demo_dir, cam_name, f"{ts:06d}.jpg")
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        return img  # raw BGR uint8

    def _resize_norm_rgb(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # TODO: undo me!
        # rgb = cv2.resize(rgb, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
        # return (rgb.astype(np.float32) / 255.0)
        return rgb

    def _rot_base_to_cam(self, R_base):
        return (T_BASE_TO_CAM_LEFT[:3,:3] @ R_base).astype(np.float32)

    def _world_to_cam3(self, p_world3):
        """Transform a single 3D point from robot base -> left camera frame."""
        ph = np.array([p_world3[0], p_world3[1], p_world3[2], 1.0], dtype=np.float64)
        pc = T_BASE_TO_CAM_LEFT @ ph
        return pc[:3]  # (Xc, Yc, Zc)

    def _row_to_action(self, row):
        # --- Right wrist position in left camera frame ---
        p_world = np.array([row["right_pos_x"], row["right_pos_y"], row["right_pos_z"]], dtype=np.float64)
        p_cam = self._world_to_cam3(p_world)  # (3,)

        # --- Right wrist orientation in left camera frame  ---
        rq = [row["right_ori_x"], row["right_ori_y"], row["right_ori_z"], row["right_ori_w"]]
        rR = R.from_quat(rq).as_matrix()
        R_cam = self._rot_base_to_cam(rR)
        ori6d = R_cam[:, :2].reshape(-1, order="F")

        # --- Right hand command joints (20) ---
        joints20 = np.asarray([row[c] for c in self.right_hand_cols], dtype=np.float32)

        # --- Final action vector: [pos_cam(3), ori6d(6), joints20(20)] = 29 dims ---
        a = np.concatenate([p_cam.astype(np.float32), ori6d.astype(np.float32), joints20], axis=0)
        return a  # (29,)

    # CHANGED: __getitem__ tail portion
    def __getitem__(self, index):
        ep_id, start_ts = self.index[index]
        demo_dir, episode_len = self.episodes[ep_id]
        t0 = min(start_ts, episode_len - 1)

        img_bgr = self._load_img_raw_bgr(demo_dir, t0, cam_name="left")
        img_bgr_left_wrist  = self._load_img_raw_bgr(demo_dir, t0, cam_name="left_wrist")
        img_bgr_right_wrist = self._load_img_raw_bgr(demo_dir, t0, cam_name="right_wrist")
        csv_path = os.path.join(demo_dir, "ee_hand.csv")
        df = pd.read_csv(csv_path)

        if self.overlay:
            row0 = df.iloc[t0]
            pts_L = self._fk_points_world(row0, "left",  kind="actual")
            pts_R = self._fk_points_world(row0, "right", kind="actual")
            img_bgr = self._project_draw_hands_on_left(img_bgr, pts_L, pts_R)

        image = self._resize_norm_rgb(img_bgr)
        wrist_image_left  = self._resize_norm_rgb(img_bgr_left_wrist)[::-1, ::-1, :]
        wrist_image_right = self._resize_norm_rgb(img_bgr_right_wrist)

        # ---------- baselines at t0 (camera-frame wrist pos+6D ori) ----------
        row0 = df.iloc[t0]
        pL0, oL0 = self._row_wrist_pose6d_in_left_cam(row0, "left")
        pR0, oR0 = self._row_wrist_pose6d_in_left_cam(row0, "right")

        # baseline CURRENT joints at t0 (20 each)
        curL0 = self._joints20_from_row(row0, "left",  kind="current")   # left_hand_*
        curR0 = self._joints20_from_row(row0, "right", kind="current")   # right_hand_*

        # ---------- actions: interleaved [L, R] per step ----------
        # per-hand action = [Δpos_cam(3), Δori6d(6), (commanded(t) − current(t0)) 20]
        tokens, action_mask = [], []
        for k in range(self.chunk_size):
            t = start_ts + k * self.stride
            if t < episode_len:
                row = df.iloc[t]

                # left wrist relative-to-t0
                pL, oL = self._row_wrist_pose6d_in_left_cam(row, "left")
                dposL  = (pL - pL0).astype(np.float32)
                doriL  = (oL - oL0).astype(np.float32)

                # left joints: commanded(t) − actual(t0)
                cmdL   = self._joints20_from_row(row, "left", kind="desired")   # left_actual_hand_*
                jabsL  = (cmdL).astype(np.float32)

                aL = np.concatenate([dposL, doriL, jabsL], axis=0)  # (29,)

                # right wrist relative-to-t0
                pR, oR = self._row_wrist_pose6d_in_left_cam(row, "right")
                dposR  = (pR - pR0).astype(np.float32)
                doriR  = (oR - oR0).astype(np.float32)

                # right joints: commanded(t) − actual(t0)
                cmdR   = self._joints20_from_row(row, "right", kind="desired")  # right_actual_hand_*
                jabsR  = (cmdR).astype(np.float32)

                aR = np.concatenate([dposR, doriR, jabsR], axis=0)  # (29,)

                tokens.extend([aL, aR])
                action_mask.extend([1.0, 1.0])
            else:
                tokens.extend([np.zeros(29, np.float32), np.zeros(29, np.float32)])
                action_mask.extend([0.0, 0.0])

        actions = np.stack(tokens, axis=0).astype(np.float32)  # (2*chunk_size, 29)

        # ---------- state (32 dims) ----------
        # [pL0(3), oL0(6), pR0(3), oR0(6), left_hand_1..7, right_hand_1..7]
        def _cmd7(row, side: str):
            cols = ([f"left_actual_hand_{i}"  for i in range(1, 8)]
                    if side == "left" else
                    [f"right_actual_hand_{i}" for i in range(1, 8)])
            return self._read_joint_array(row, cols)  # (7,)

        cmd7L = _cmd7(row0, "left")
        cmd7R = _cmd7(row0, "right")

        state = np.concatenate([pL0, oL0, pR0, oR0, cmd7L, cmd7R], axis=0).astype(np.float32)  # (32,)

        return {
            "image": image.astype(np.uint8),
            "wrist_image_left":  wrist_image_left.astype(np.uint8),
            "wrist_image_right": wrist_image_right.astype(np.uint8),
            "state": state,                 # (32,)
            "actions": actions,             # (2*chunk_size, 29)
            "task": self.task,
        }

# ===================== Simple visualization =====================

# def _cam3_to_uv(Pc, K, H, W):
#     Z = float(Pc[2])
#     if Z <= 1e-6:
#         return None
#     u = int(K[0,0] * Pc[0] / Z + K[0,2])
#     v = int(K[1,1] * Pc[1] / Z + K[1,2])
#     if 0 <= u < W and 0 <= v < H:
#         return (u, v)
#     return None

# def _draw_points(img_bgr, pts_uv, color=(0, 200, 255), radius=5, label=None):
#     for (uv, name) in pts_uv:
#         if uv is None:
#             continue
#         cv2.circle(img_bgr, uv, radius, color, -1, cv2.LINE_AA)
#         if label:
#             cv2.putText(img_bgr, f"{label}:{name}", (uv[0]+3, uv[1]-3),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 2, cv2.LINE_AA)
#             cv2.putText(img_bgr, f"{label}:{name}", (uv[0]+3, uv[1]-3),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)

# def save_visualizations(ds, out_dir, num_samples=4):
#     os.makedirs(out_dir, exist_ok=True)
#     dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

#     # names in order: thumb, index, middle, ring, pinky
#     TIP_LINKS_L = [f"ll_dg_{i}_tip" for i in [1,2,3,4,5]]
#     TIP_LINKS_R = [f"rl_dg_{i}_tip" for i in [1,2,3,4,5]]

#     for i, batch in enumerate(dl):
#         # raw image at this index (load again to keep original size)
#         ep_id, t0 = ds.index[0] if i == 0 else ds.index[i]
#         demo_dir, _ = ds.episodes[ep_id]
#         img_bgr = ds._load_img_raw_bgr(demo_dir, t0, cam_name="left")
#         H, W = img_bgr.shape[:2]

#         # TODO: integrate into the core code above in getitem
#         K_use = scale_K(K_LEFT, new_W=W, new_H=H, orig_W=1280, orig_H=720)


#         # row for projections
#         csv_path = os.path.join(demo_dir, "ee_hand.csv")
#         row0 = pd.read_csv(csv_path).iloc[t0]

#         # ----- ACTIONS VIS -----
#         pts_uv_L, pts_uv_R = [], []

#         # wrists
#         if ds.hand_mode in ("left", "both"):
#             pL, _ = ds._row_wrist_pose6d_in_left_cam(row0, "left")
#             uvL = _cam3_to_uv(pL, K_use, H, W)
#             pts_uv_L.append((uvL, "wrist"))

#         if ds.hand_mode in ("right", "both"):
#             pR, _ = ds._row_wrist_pose6d_in_left_cam(row0, "right")
#             uvR = _cam3_to_uv(pR, K_use, H, W)
#             pts_uv_R.append((uvR, "wrist"))

#         # fingertips
#         if ds.hand_mode in ("left", "both"):
#             fkL = ds._fk_points_world(row0, "left", kind="actual")
#             for name in TIP_LINKS_L:
#                 # print(name, fkL[name])
#                 Pc = ds._world_to_cam3(fkL[name]) if name in fkL else None
#                 # print("L tip", name, "Pc", Pc)
#                 pts_uv_L.append((_cam3_to_uv(Pc, K_use, H, W) if Pc is not None else None,
#                                  name.split("_")[-1]))

#         if ds.hand_mode in ("right", "both"):
#             fkR = ds._fk_points_world(row0, "right", kind="actual")
#             for name in TIP_LINKS_R:
#                 Pc = ds._world_to_cam3(fkR[name]) if name in fkR else None
#                 pts_uv_R.append((_cam3_to_uv(Pc, K_use, H, W) if Pc is not None else None,
#                                  name.split("_")[-1]))

#         img_actions = img_bgr.copy()
#         _draw_points(img_actions, pts_uv_L, color=(255, 160, 0),  radius=6, label="L")
#         _draw_points(img_actions, pts_uv_R, color=(0, 220, 0),    radius=6, label="R")

#         out_actions = os.path.join(out_dir, f"sample_{i:02d}_actions.png")
#         cv2.imwrite(out_actions, img_actions)

#         # ----- STATE VIS -----
#         # both-mode: state uses L/R wrist + L/R thumb+index.
#         # left/right-mode: show the same thumb+index for the active hand for consistency.
#         img_state = img_bgr.copy()
#         pts_state = []

#         # L wrist + thumb/index
#         if ds.hand_mode in ("left", "both"):
#             pL, _ = ds._row_wrist_pose6d_in_left_cam(row0, "left")
#             pts_state.append((_cam3_to_uv(pL, K_use, H, W), "L:wrist"))
#             if 'fkL' not in locals():
#                 fkL = ds._fk_points_world(row0, "left", kind="actual")
#             for tip_link, tip_name in [(TIP_LINKS_L[0], "L:thumb"), (TIP_LINKS_L[1], "L:index")]:
#                 Pc = ds._world_to_cam3(fkL[tip_link]) if tip_link in fkL else None
#                 pts_state.append((_cam3_to_uv(Pc, K_use, H, W) if Pc is not None else None, tip_name))

#         # R wrist + thumb/index
#         if ds.hand_mode in ("right", "both"):
#             pR, _ = ds._row_wrist_pose6d_in_left_cam(row0, "right")
#             pts_state.append((_cam3_to_uv(pR, K_use, H, W), "R:wrist"))
#             if 'fkR' not in locals():
#                 fkR = ds._fk_points_world(row0, "right", kind="actual")
#             for tip_link, tip_name in [(TIP_LINKS_R[0], "R:thumb"), (TIP_LINKS_R[1], "R:index")]:
#                 Pc = ds._world_to_cam3(fkR[tip_link]) if tip_link in fkR else None
#                 pts_state.append((_cam3_to_uv(Pc, K_use, H, W) if Pc is not None else None, tip_name))

#         # draw with two colors for readability
#         _draw_points(img_state, [(uv, n) for (uv, n) in pts_state if n.startswith("L")],
#                      color=(255, 160, 0), radius=6, label=None)
#         _draw_points(img_state, [(uv, n) for (uv, n) in pts_state if n.startswith("R")],
#                      color=(0, 220, 0), radius=6, label=None)

#         out_state = os.path.join(out_dir, f"sample_{i:02d}_state.png")
#         cv2.imwrite(out_state, img_state)

#         print(f"saved\n  {out_actions}\n  {out_state}")
#         if i + 1 >= num_samples:
#             break


# ----------------- quick runner -----------------
# if __name__ == "__main__":
#     dataset_root = "/iris/projects/humanoid/dataset/DEMO_QUEST_CONTROLLER/QUEST_ASSEMBLE_ROBOT"  # change as needed
#     # dataset_root = "/iris/projects/humanoid/tesollo_dataset/robot_data_0903/red_cube_inbox"  # change as needed


    
#     ds = GalaxeaDatasetKeypointsJoints(
#         dataset_dir=dataset_root,
#         chunk_size=8,
#         stride=3,
#         overlay=False,
#         hand_mode="both",  # "left" | "right" | "both"
#     )
#     save_visualizations(ds, out_dir="/iris/projects/humanoid/openpi/robot_vis", num_samples=500)

#################################################################
#################################################################
# UNCOMMENT ME to visualize the dataset!
#################################################################
#################################################################

import os
import numpy as np
import cv2
from torch.utils.data import DataLoader

# ---- import your dataset class first ----
# from your_file import GalaxeaDatasetKeypointsJoints

def save_rgb01(path, rgb01):
    """rgb01: (H,W,3) float32 in [0,1]"""
    bgr_u8 = rgb01.astype(np.uint8)[:, :, ::-1]  # RGB->BGR
    cv2.imwrite(path, bgr_u8)

if __name__ == "__main__":
    # dataset_root = "/iris/projects/humanoid/dataset/DEMO_QUEST_CONTROLLER/QUEST_ASSEMBLE_ROBOT"  # change as needed
    dataset_root = "/iris/projects/humanoid/dataset/DEMO_PICK_PLACE/banana"
    # dataset_root = "/iris/projects/humanoid/tesollo_dataset/robot_data_0903/red_cube_inbox"  # change if needed
    # dataset_root = "/iris/projects/humanoid/dataset/New_QUEST_DATA_ROBOT"

    ds = GalaxeaDatasetKeypointsJoints(
        dataset_dir=dataset_root,
        chunk_size=8,
        stride=3,
        overlay=True,   # turn on drawing
        task="vertical_pick_place"
        # hand_mode="both",  # "left" | "right" | "both"
    )

    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    out_dir = "/iris/projects/humanoid/openpi/robot_vis"
    os.makedirs(out_dir, exist_ok=True)

    for i, batch in enumerate(dl):
        # tensors -> numpy
        img_main = batch["image"][0].numpy()                 # (H,W,3) float32 [0,1]
        img_wl   = batch["wrist_image_left"][0].numpy()      # (H,W,3) float32 [0,1]
        img_wr   = batch["wrist_image_right"][0].numpy()     # (H,W,3) float32 [0,1]

        state   = batch["state"][0].numpy()
        actions = batch["actions"][0].numpy()
        task    = batch["task"][0]

        # save individually
        save_rgb01(os.path.join(out_dir, f"sample_{i:02d}.png"),           img_main)
        # save_rgb01(os.path.join(out_dir, f"sample_{i:02d}_wrist_left.png"),  img_wl)
        # save_rgb01(os.path.join(out_dir, f"sample_{i:02d}_wrist_right.png"), img_wr)

        # optional: quick strip for visual parity check
        try:
            strip = np.concatenate([img_main, img_wl, img_wr], axis=1)  # (H, 3W, 3)
            # save_rgb01(os.path.join(out_dir, f"sample_{i:02d}_strip.png"), strip)
        except Exception as e:
            print(f"[warn] could not make strip for sample {i}: {e}")

        print(
            f"saved sample_{i:02d} main/wrist images  |  "
            f"state {state.shape}  actions {actions.shape}  task={task}"
        )

        if i >= 10:
            break


#########################################################################
#########################################################################
# Below is for visualizing the actions and the states projected onto the image
#########################################################################
#########################################################################

# save as: visualize_galaxea_samples.py
# import os, cv2, numpy as np, argparse, torch
# from scipy.spatial.transform import Rotation as R

# # --- intrinsics + scaler (match your dataset module) ---
# K_LEFT = np.array([[730.2571411132812, 0.0, 637.2598876953125],
#                    [0.0, 730.2571411132812, 346.41082763671875],
#                    [0.0, 0.0, 1.0]], dtype=np.float64)

# def scale_K(K, new_W, new_H, orig_W=1280, orig_H=720):
#     K = K.copy()
#     K[0,0] *= new_W / orig_W   # fx
#     K[1,1] *= new_H / orig_H   # fy

#     K[0,2] *= new_W / orig_W   # cx
#     K[1,2] *= new_H / orig_H   # cy
#     return K

# # --- projection helpers ---
# def cam3_to_uv(points_cam, K, W, H):
#     """
#     points_cam: (N,3) array of [Xc,Yc,Zc] in camera frame.
#     Returns: list of (u,v) or None (length N).
#     """
#     pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
#     fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
#     out = []
#     for Xc, Yc, Zc in pts:
#         if not np.isfinite(Zc) or Zc <= 1e-6:
#             out.append(None); continue
#         u = int(fx * Xc / Zc + cx)
#         v = int(fy * Yc / Zc + cy)
#         out.append((u, v) if (0 <= u < W and 0 <= v < H) else None)
#     return out

# def draw_points(img_bgr, uvs, color=(0,255,0), r=5, label_prefix=None):
#     for i, uv in enumerate(uvs):
#         if uv is None: continue
#         cv2.circle(img_bgr, uv, r, color, -1, cv2.LINE_AA)
#         if label_prefix is not None:
#             cv2.putText(img_bgr, f"{label_prefix}{i}", (uv[0]+3, uv[1]-3),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 2, cv2.LINE_AA)
#             cv2.putText(img_bgr, f"{label_prefix}{i}", (uv[0]+3, uv[1]-3),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)

# def draw_trajectory(img_bgr, uvs, color, r=3, thickness=2):
#     prev = None
#     for uv in uvs:
#         if uv is not None:
#             cv2.circle(img_bgr, uv, r, color, -1, cv2.LINE_AA)
#             if prev is not None:
#                 cv2.line(img_bgr, prev, uv, color, thickness, cv2.LINE_AA)
#             prev = uv
#         else:
#             prev = None

# def to_bgr_uint8(image_float_rgb):
#     img = (np.clip(image_float_rgb, 0, 1) * 255).astype(np.uint8)
#     return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset_dir", required=True, help="Path to demos root")
#     parser.add_argument("--out_dir", required=True, help="Output folder for visualizations")
#     parser.add_argument("--num_samples", type=int, default=8)
#     parser.add_argument("--chunk_size", type=int, default=4)
#     parser.add_argument("--stride", type=int, default=3)
#     parser.add_argument("--overlay", action="store_true", help="If your class draws FK overlays")
#     args = parser.parse_args()
#     os.makedirs(args.out_dir, exist_ok=True)

#     # import your dataset class from your module
#     from openpi.training.galaxea_dataset import GalaxeaDatasetKeypointsJoints  # adjust path if needed

#     ds = GalaxeaDatasetKeypointsJoints(
#         dataset_dir=args.dataset_dir,
#         chunk_size=args.chunk_size,
#         stride=args.stride,
#         overlay=args.overlay,
#         task = "placeholder"
#     )
#     # no loader batching needed; just iterate
#     n = min(args.num_samples, len(ds))

#     for i in range(n):
#         item = ds[i]
#         img_rgb = item["image"]        # float [0,1], shape (H,W,3)
#         H, W = img_rgb.shape[:2]
#         K = scale_K(K_LEFT, W, H, orig_W=1280, orig_H=720)
#         img_bgr_actions = to_bgr_uint8(img_rgb)
#         img_bgr_state   = to_bgr_uint8(img_rgb)

#         # ---------- ACTIONS ----------
#         # actions: shape (2*chunk, 24). First two rows are L then R for t0.
#         actions = item["actions"]  # (2*chunk_size, 29)
#         s = item["state"]          # (32,) = [pL0(3), oL0(6), pR0(3), oR0(6), cmd7L(7), cmd7R(7)]

#         # baseline wrists at t0 from state
#         pL0 = s[0:3].astype(np.float32)
#         pR0 = s[9:12].astype(np.float32)

#         aL = actions[0::2, 0:3]    # (T,3)  Δpos_cam for left
#         aR = actions[1::2, 0:3]    # (T,3)  Δpos_cam for right
#         pL = pL0[None, :] + aL     # (T,3) absolute left
#         pR = pR0[None, :] + aR     # (T,3) absolute right

#         uvs_L = cam3_to_uv(pL, K, W, H)
#         uvs_R = cam3_to_uv(pR, K, W, H)

#         draw_trajectory(img_bgr_actions, uvs_L, color=(0,255,0))
#         draw_trajectory(img_bgr_actions, uvs_R, color=(255,0,0))

#         # optional: highlight start/end waypoints
#         for uv in (uvs_L[0], uvs_L[-1]):
#             if uv is not None: cv2.circle(img_bgr_actions, uv, 6, (0,200,0), -1, cv2.LINE_AA)
#         for uv in (uvs_R[0], uvs_R[-1]):
#             if uv is not None: cv2.circle(img_bgr_actions, uv, 6, (200,0,0), -1, cv2.LINE_AA)
        
#         # ---------- STATE ----------
#         # state (expected 30): [pL(3), oL(6), pR(3), oR(6), L_thumb(3), L_index(3), R_thumb(3), R_index(3)]
#         s = item["state"]
#         pts_cam_state = []
#         ok_state = s.ndim == 1 and s.size >= 30
#         if ok_state:
#             pL_s = s[0:3]; pR_s = s[9:12]
#             # L_thumb = s[18:21]; L_index = s[21:24]
#             # R_thumb = s[24:27]; R_index = s[27:30]
#             # pts_cam_state += [pL_s, pR_s, L_thumb, L_index, R_thumb, R_index]
#             pts_cam_state += [pL_s, pR_s]

#             uvs_state = cam3_to_uv(pts_cam_state, K, W, H)
#             # draw: wrists larger, fingertips smaller
#             draw_points(img_bgr_state, [uvs_state[0]], color=(0,255,0), r=7, label_prefix="Lw")
#             draw_points(img_bgr_state, [uvs_state[1]], color=(255,0,0), r=7, label_prefix="Rw")
#             # draw_points(img_bgr_state, uvs_state[2:4], color=(0,200,255), r=5, label_prefix="L")
#             # draw_points(img_bgr_state, uvs_state[4:6], color=(255,200,0), r=5, label_prefix="R")

#         # ---------- save ----------
#         out_actions = os.path.join(args.out_dir, f"sample_{i:03d}_actions.jpg")
#         out_state   = os.path.join(args.out_dir, f"sample_{i:03d}_state.jpg")
#         cv2.imwrite(out_actions, img_bgr_actions)
#         cv2.imwrite(out_state,   img_bgr_state)
#         print(f"[saved] {out_actions}  |  {out_state}")

# if __name__ == "__main__":
#     main()
