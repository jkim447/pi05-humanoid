"""
usage:
uv run src/openpi/training/galaxea_dataset.py
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
ROBOT_TO_MANO = {
    # thumb
    "rl_thumb_knuckle1": "rightThumbKnuckle",
    "rl_thumb_knuckle2": "rightThumbIntermediateBase",
    "rl_thumb_tip":      "rightThumbTip",
    # index
    "rl_index_knuckle1": "rightIndexFingerKnuckle",
    "rl_index_knuckle2": "rightIndexFingerIntermediateBase",
    "rl_index_tip":      "rightIndexFingerTip",
    # middle
    "rl_middle_knuckle1": "rightMiddleFingerKnuckle",
    "rl_middle_knuckle2": "rightMiddleFingerIntermediateBase",
    "rl_middle_tip":      "rightMiddleFingerTip",
    # ring
    "rl_ring_knuckle1": "rightRingFingerKnuckle",
    "rl_ring_knuckle2": "rightRingFingerIntermediateBase",
    "rl_ring_tip":      "rightRingFingerTip",
    # pinky
    "rl_pinky_knuckle1": "rightLittleFingerKnuckle",
    "rl_pinky_knuckle2": "rightLittleFingerIntermediateBase",
    "rl_pinky_tip":      "rightLittleFingerTip",
}

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
ROBOT_TO_MANO = {WRIST: "rightHand"}
for mano_finger, chain in FINGERS_ROBOT.items():
    segs = ["Knuckle","IntermediateBase","IntermediateTip","Tip"]
    for seg, link in zip(segs, chain):
        ROBOT_TO_MANO[link] = f"right{mano_finger}{seg}"

def _link_to_mano(name: str) -> str | None:
    # 1) explicit mapping wins
    if name in ROBOT_TO_MANO:
        return ROBOT_TO_MANO[name]

    # 2) heuristic: rl_<finger>_<part>
    base = name[3:] if name.startswith("rl_") else name  # e.g. "index_tip"
    parts = base.split("_")
    if len(parts) < 2:
        return None

    finger_raw, part = parts[0], "_".join(parts[1:])  # e.g. "index", "knuckle2"
    finger_map = {
        "thumb":  "Thumb",
        "index":  "IndexFinger",
        "middle": "MiddleFinger",
        "ring":   "RingFinger",
        "pinky":  "LittleFinger",
        "little": "LittleFinger",
    }
    if finger_raw not in finger_map:
        return None

    # map part -> MANO segment
    # knuckle1 -> Knuckle
    # knuckle2 -> IntermediateBase
    # knuckle3 -> IntermediateTip
    # tip      -> Tip
    part_map = {
        "knuckle1": "Knuckle",
        "knuckle2": "IntermediateBase",
        "knuckle3": "IntermediateTip",
        "tip":      "Tip",
    }
    seg = part_map.get(part)
    if not seg:
        return None

    return f"right{finger_map[finger_raw]}{seg}"

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
# block transfer data
T_BASE_TO_CAM_LEFT = np.linalg.inv(np.array([
    [ 0.00692993, -0.87310148,  0.48748926,  0.14062141],
    [-0.99995006, -0.00956093, -0.00290894,  0.03612369],
    [ 0.00720065, -0.48744476, -0.87312414,  0.46063114],
    [ 0., 0., 0., 1. ]
], dtype=np.float64))


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
    def __init__(self, dataset_dir, chunk_size, stride=3, overlay=False,
                 hand_mode: str = "right",  # "left", "right", or "both"
                 urdf_left_path="/iris/projects/humanoid/act/dg_description/urdf/dg5f_left.urdf",
                 urdf_right_path="/iris/projects/humanoid/act/dg_description/urdf/dg5f_right.urdf"):
        super(GalaxeaDatasetKeypointsJoints).__init__()
        self.dataset_dir = dataset_dir
        self.chunk_size = chunk_size
        self.img_height = 224
        self.img_width  = 224
        self.stride = stride
        self.overlay = overlay
        self.hand_mode = hand_mode  # NEW

        # columns
        self.left_hand_cols  = [f"left_hand_{i}"  for i in range(20)]
        self.right_hand_cols = [f"right_hand_{i}" for i in range(20)]
        self.left_actual_hand_cols  = [f"left_actual_hand_{i}"  for i in range(20)]
        self.right_actual_hand_cols = [f"right_actual_hand_{i}" for i in range(20)]

        self.action_camera = "left"

        # CHANGED: load both URDFs when needed
        if self.overlay or (self.hand_mode in ("left", "both")):
            self.robot_l = URDF.load(urdf_left_path)
            self.left_joint_names = [j.name for j in self.robot_l.joints if j.joint_type != "fixed"]
        if self.overlay or (self.hand_mode in ("right", "both")):
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

        # Build flat index of (ep_id, t0) for all valid starting timesteps
        # We require full chunks (no padding). If you want padding, see the variant below.
        horizon = self.chunk_size * self.stride
        self.index = []
        for ep_id, (_, N) in enumerate(self.episodes):
            # print("going thru:", ep_id)
            last_start = N - horizon
            if last_start < 0:
                continue
            # t0 = 0, stride, 2*stride, ..., <= last_start
            self.index.extend((ep_id, t0) for t0 in range(0, last_start + 1, self.stride))

    def __len__(self):
        return len(self.index)

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
        cols = primary if all(c in row for c in primary) else fallback
        return [float(row[c]) for c in cols]

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
        else:
            T_ee = _pose_to_T([row["right_pos_x"], row["right_pos_y"], row["right_pos_z"]],
                            [row["right_ori_x"], row["right_ori_y"], row["right_ori_z"], row["right_ori_w"]])
            T_hand = T_ee @ T_EE_TO_HAND_R
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
    def _row_to_hand_action24(self, row, side: str) -> np.ndarray:
        p_cam, ori6d = self._row_wrist_pose6d_in_left_cam(row, side)
        # Actions should reflect commanded hand → use kind="cmd"
        pts_map_cmd = self._fk_points_world(row, side, kind="cmd")
        tip15 = self._tip_positions_in_left_cam(pts_map_cmd, side)  # 5 tips x 3
        return np.concatenate([p_cam, ori6d, tip15], axis=0)  # (24,)

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
            pt_radius=6,
            line_thickness=3,
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
        rgb = cv2.resize(rgb, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
        return (rgb.astype(np.float32) / 255.0)

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

        img_bgr = self._load_img_raw_bgr(demo_dir, start_ts, cam_name="left")
        csv_path = os.path.join(demo_dir, "ee_hand.csv")
        df = pd.read_csv(csv_path)

        if self.overlay:
            row0 = df.iloc[start_ts]
            pts_L = pts_R = None
            if self.hand_mode in ("left", "both"):
                pts_L = self._fk_points_world(row0, "left", kind="actual")
                if not pts_L:  # fallback
                    pts_L = self._fk_points_world(row0, "left", kind="cmd")
            if self.hand_mode in ("right", "both"):
                pts_R = self._fk_points_world(row0, "right", kind="actual")
                if not pts_R:
                    pts_R = self._fk_points_world(row0, "right", kind="cmd")
            img_bgr = self._project_draw_hands_on_left(img_bgr, pts_L, pts_R)
        image = self._resize_norm_rgb(img_bgr)

        end_ts = start_ts + self.chunk_size * self.stride
        rows = [df.iloc[t] for t in range(start_ts, end_ts, self.stride)]

        # Build per-time-step tokens
        tokens = []  # will be [L,R,L,R,...] depending on mode
        if self.hand_mode == "left":
            for row in rows:
                aL = self._row_to_hand_action24(row, "left")
                tokens.append(aL)                 # even index -> left
                tokens.append(np.zeros(24, np.float32))  # odd index -> zeros
            state = self._row_to_hand_action24(df.iloc[start_ts], "left")  # 24D

        elif self.hand_mode == "right":
            for row in rows:
                tokens.append(np.zeros(24, np.float32))  # even index -> zeros
                aR = self._row_to_hand_action24(row, "right")
                tokens.append(aR)                 # odd index -> right
            state = self._row_to_hand_action24(df.iloc[start_ts], "right")  # 24D

        else:  # "both"
            for row in rows:
                aL = self._row_to_hand_action24(row, "left")
                aR = self._row_to_hand_action24(row, "right")
                tokens.append(aL)  # even -> left
                tokens.append(aR)  # odd  -> right

            # 30D state: L wrist(3)+ori6(6) + R wrist(3)+ori6(6) + L tips {thumb,index}(6) + R tips {thumb,index}(6)
            row0 = df.iloc[start_ts]
            pL, oL = self._row_wrist_pose6d_in_left_cam(row0, "left")
            pR, oR = self._row_wrist_pose6d_in_left_cam(row0, "right")
            ptsL = self._fk_points_world(row0, "left", kind="actual")
            ptsR = self._fk_points_world(row0, "right", kind="actual")
            # thumb = index 0, index = index 1 in TIP_LINKS_*
            L_thumb = self._world_to_cam3(ptsL[TIP_LINKS_L[0]]) if TIP_LINKS_L[0] in ptsL else np.zeros(3, np.float32)
            L_index = self._world_to_cam3(ptsL[TIP_LINKS_L[1]]) if TIP_LINKS_L[1] in ptsL else np.zeros(3, np.float32)
            R_thumb = self._world_to_cam3(ptsR[TIP_LINKS_R[0]]) if TIP_LINKS_R[0] in ptsR else np.zeros(3, np.float32)
            R_index = self._world_to_cam3(ptsR[TIP_LINKS_R[1]]) if TIP_LINKS_R[1] in ptsR else np.zeros(3, np.float32)

            state = np.concatenate([
                pL.astype(np.float32), oL.astype(np.float32),
                pR.astype(np.float32), oR.astype(np.float32),
                L_thumb.astype(np.float32), L_index.astype(np.float32),
                R_thumb.astype(np.float32), R_index.astype(np.float32),
            ], axis=0)  # (30,)

        actions = np.stack(tokens, axis=0).astype(np.float32)  # (2*chunk_size, 24)

        # state shape varies by mode
        return {
            "image":   image.astype(np.float32),     # (H, W, 3) in [0,1]
            "state":   state.astype(np.float32),     # (24,) for L/R, (30,) for both
            "actions": actions,                      # (2*chunk_size, 24), interleaved
            "task":    "place red cube into box",
        }


# ===================== Simple visualization =====================

def _cam3_to_uv(Pc, K, H, W):
    Z = float(Pc[2])
    if Z <= 1e-6:
        return None
    u = int(K[0,0] * Pc[0] / Z + K[0,2])
    v = int(K[1,1] * Pc[1] / Z + K[1,2])
    if 0 <= u < W and 0 <= v < H:
        return (u, v)
    return None

def _draw_points(img_bgr, pts_uv, color=(0, 200, 255), radius=5, label=None):
    for (uv, name) in pts_uv:
        if uv is None:
            continue
        cv2.circle(img_bgr, uv, radius, color, -1, cv2.LINE_AA)
        if label:
            cv2.putText(img_bgr, f"{label}:{name}", (uv[0]+3, uv[1]-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(img_bgr, f"{label}:{name}", (uv[0]+3, uv[1]-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)

def save_visualizations(ds, out_dir, num_samples=4):
    os.makedirs(out_dir, exist_ok=True)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    # names in order: thumb, index, middle, ring, pinky
    TIP_LINKS_L = [f"ll_dg_{i}_tip" for i in [1,2,3,4,5]]
    TIP_LINKS_R = [f"rl_dg_{i}_tip" for i in [1,2,3,4,5]]

    for i, batch in enumerate(dl):
        # raw image at this index (load again to keep original size)
        ep_id, t0 = ds.index[0] if i == 0 else ds.index[i]
        demo_dir, _ = ds.episodes[ep_id]
        img_bgr = ds._load_img_raw_bgr(demo_dir, t0, cam_name="left")
        H, W = img_bgr.shape[:2]

        # TODO: integrate into the core code above in getitem
        K_use = scale_K(K_LEFT, new_W=W, new_H=H, orig_W=1280, orig_H=720)


        # row for projections
        csv_path = os.path.join(demo_dir, "ee_hand.csv")
        row0 = pd.read_csv(csv_path).iloc[t0]

        # ----- ACTIONS VIS -----
        pts_uv_L, pts_uv_R = [], []

        # wrists
        if ds.hand_mode in ("left", "both"):
            pL, _ = ds._row_wrist_pose6d_in_left_cam(row0, "left")
            uvL = _cam3_to_uv(pL, K_use, H, W)
            pts_uv_L.append((uvL, "wrist"))

        if ds.hand_mode in ("right", "both"):
            pR, _ = ds._row_wrist_pose6d_in_left_cam(row0, "right")
            uvR = _cam3_to_uv(pR, K_use, H, W)
            pts_uv_R.append((uvR, "wrist"))

        # fingertips
        if ds.hand_mode in ("left", "both"):
            fkL = ds._fk_points_world(row0, "left", kind="cmd")
            for name in TIP_LINKS_L:
                # print(name, fkL[name])
                Pc = ds._world_to_cam3(fkL[name]) if name in fkL else None
                # print("L tip", name, "Pc", Pc)
                pts_uv_L.append((_cam3_to_uv(Pc, K_use, H, W) if Pc is not None else None,
                                 name.split("_")[-1]))

        if ds.hand_mode in ("right", "both"):
            fkR = ds._fk_points_world(row0, "right", kind="cmd")
            for name in TIP_LINKS_R:
                Pc = ds._world_to_cam3(fkR[name]) if name in fkR else None
                pts_uv_R.append((_cam3_to_uv(Pc, K_use, H, W) if Pc is not None else None,
                                 name.split("_")[-1]))

        img_actions = img_bgr.copy()
        _draw_points(img_actions, pts_uv_L, color=(255, 160, 0),  radius=6, label="L")
        _draw_points(img_actions, pts_uv_R, color=(0, 220, 0),    radius=6, label="R")

        out_actions = os.path.join(out_dir, f"sample_{i:02d}_actions.png")
        cv2.imwrite(out_actions, img_actions)

        # ----- STATE VIS -----
        # both-mode: state uses L/R wrist + L/R thumb+index.
        # left/right-mode: show the same thumb+index for the active hand for consistency.
        img_state = img_bgr.copy()
        pts_state = []

        # L wrist + thumb/index
        if ds.hand_mode in ("left", "both"):
            pL, _ = ds._row_wrist_pose6d_in_left_cam(row0, "left")
            pts_state.append((_cam3_to_uv(pL, K_use, H, W), "L:wrist"))
            if 'fkL' not in locals():
                fkL = ds._fk_points_world(row0, "left", kind="actual")
            for tip_link, tip_name in [(TIP_LINKS_L[0], "L:thumb"), (TIP_LINKS_L[1], "L:index")]:
                Pc = ds._world_to_cam3(fkL[tip_link]) if tip_link in fkL else None
                pts_state.append((_cam3_to_uv(Pc, K_use, H, W) if Pc is not None else None, tip_name))

        # R wrist + thumb/index
        if ds.hand_mode in ("right", "both"):
            pR, _ = ds._row_wrist_pose6d_in_left_cam(row0, "right")
            pts_state.append((_cam3_to_uv(pR, K_use, H, W), "R:wrist"))
            if 'fkR' not in locals():
                fkR = ds._fk_points_world(row0, "right", kind="actual")
            for tip_link, tip_name in [(TIP_LINKS_R[0], "R:thumb"), (TIP_LINKS_R[1], "R:index")]:
                Pc = ds._world_to_cam3(fkR[tip_link]) if tip_link in fkR else None
                pts_state.append((_cam3_to_uv(Pc, K_use, H, W) if Pc is not None else None, tip_name))

        # draw with two colors for readability
        _draw_points(img_state, [(uv, n) for (uv, n) in pts_state if n.startswith("L")],
                     color=(255, 160, 0), radius=6, label=None)
        _draw_points(img_state, [(uv, n) for (uv, n) in pts_state if n.startswith("R")],
                     color=(0, 220, 0), radius=6, label=None)

        out_state = os.path.join(out_dir, f"sample_{i:02d}_state.png")
        cv2.imwrite(out_state, img_state)

        print(f"saved\n  {out_actions}\n  {out_state}")
        if i + 1 >= num_samples:
            break


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
    bgr_u8 = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]  # RGB->BGR
    cv2.imwrite(path, bgr_u8)

if __name__ == "__main__":
#     dataset_root = "/iris/projects/humanoid/dataset/DEMO_QUEST_CONTROLLER/QUEST_ASSEMBLE_ROBOT"  # change as needed

    dataset_root = "/iris/projects/humanoid/tesollo_dataset/robot_data_0903/red_cube_inbox"  # change if needed

    ds = GalaxeaDatasetKeypointsJoints(
        dataset_dir=dataset_root,
        chunk_size=8,
        stride=3,
        overlay=True,   # turn on drawing
    )

    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    out_dir = "/iris/projects/humanoid/openpi/robot_vis"
    os.makedirs(out_dir, exist_ok=True)

    # grab and save first 8 samples
    for i, batch in enumerate(dl):
        img = batch["image"][0].numpy()          # (H,W,3) float32 [0,1]
        state = batch["state"][0].numpy()        # (32,)
        actions = batch["actions"][0].numpy()    # (2*chunk, 32)
        task = batch["task"][0]

        save_rgb01(os.path.join(out_dir, f"sample_{i:02d}.png"), img)
        print(f"saved {os.path.join(out_dir, f'sample_{i:02d}.png')}  |  state {state.shape}  actions {actions.shape}  task={task}")

        if i >= 7:
            break

    print(f"Done. Check images in: {os.path.abspath(out_dir)}")