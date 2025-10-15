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
T_BASE_TO_CAM_LEFT = np.linalg.inv(np.array([
    [ 0.00692993, -0.87310148,  0.48748926,  0.14062141],
    [-0.99995006, -0.00956093, -0.00290894,  0.03612369],
    [ 0.00720065, -0.48744476, -0.87312414,  0.46063114],
    [ 0., 0., 0., 1. ]
], dtype=np.float64))

# similar as above for the right camera
T_BASE_TO_CAM_RIGHT = np.linalg.inv(np.array([
    [-0.00334115, -0.8768872 ,  0.48068458,  0.14700305],
    [-0.99996141,  0.0068351 ,  0.00551836, -0.02680847],
    [-0.00812451, -0.4806476 , -0.87687621,  0.46483729],
    [ 0., 0., 0., 1. ]
], dtype=np.float64))

class GalaxeaDatasetKeypointsJoints(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, chunk_size, stride = 3, overlay=False,
        ):
        super(GalaxeaDatasetKeypointsJoints).__init__()
        self.dataset_dir = dataset_dir
        self.chunk_size = chunk_size
        self.img_height = 224
        self.img_width  = 224
        self.stride = stride
        self.overlay = overlay  # whether to overlay keypoints
        urdf_right_path = urdf_right_path="/iris/projects/humanoid/act/dg_description/urdf/dg5f_right.urdf"
      
        self.right_hand_cols = [f"right_hand_{i}" for i in range(20)]
        self.right_actual_hand_cols = [f"right_actual_hand_{i}" for i in range(20)]
        self.action_camera = "left"  # which camera to use for action ref frame

        # Load URDF + connections only when we actually overlay
        if self.overlay:
            self.robot_r = URDF.load(urdf_right_path)
            self.right_joint_names = [j.name for j in self.robot_r.joints if j.joint_type != "fixed"]
            # draw skeleton connections (right)
            self.right_connections = [(j.parent, j.child) for j in self.robot_r.joints]
            # connect fingertips -> mids -> knuckles for nicer lines
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


    def _project_draw_right_on_left(self, img_bgr, pts_map):
        """
        Occlusion-aware drawing of the robot right hand using MANO-style colors.
        """
        h, w = img_bgr.shape[:2]
        fx, fy, cx, cy = K_LEFT[0,0], K_LEFT[1,1], K_LEFT[0,2], K_LEFT[1,2]

        # ---- ordered names (wrist + 5×4 fingers) ----
        names = [WRIST] + [n for chain in FINGERS_ROBOT.values() for n in chain]

        uv, z = [], []
        for name in names:
            if name not in pts_map:
                uv.append(None); z.append(np.inf)
                continue
            x, y, z0 = pts_map[name]
            Pc = T_BASE_TO_CAM_LEFT @ np.array([x, y, z0, 1.0], dtype=np.float64)
            Z = float(Pc[2])
            if Z <= 1e-6:
                uv.append(None); z.append(np.inf)
                continue
            u = int(fx * Pc[0] / Z + cx)
            v = int(fy * Pc[1] / Z + cy)
            if 0 <= u < w and 0 <= v < h:
                uv.append((u, v)); z.append(Z)
            else:
                uv.append(None); z.append(Z)
        z = np.array(z, dtype=np.float32)

        # ---- edges (wrist→knuckle + per-finger chains) ----
        edges_by_name = []
        for chain in FINGERS_ROBOT.values():
            edges_by_name.append((WRIST, chain[0]))
            edges_by_name.extend(zip(chain, chain[1:]))

        # ---- per-joint colors via MANO mapping ----
        fallback = (210,210,210)
        color_of = {}
        for n in names:
            mano = _link_to_mano(n)
            color_of[n] = JOINT_COLOR_BGR.get(mano, fallback) if mano else fallback

        # ---- convert BGR→RGB float and draw ----
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
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

        # ---- back to BGR for later pipeline ----
        return cv2.cvtColor((img_rgb*255).astype(np.uint8), cv2.COLOR_RGB2BGR)


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

    def __getitem__(self, index):
        ep_id, start_ts = self.index[index]
        demo_dir, episode_len = self.episodes[ep_id]

        # image at the window start
        img_bgr = self._load_img_raw_bgr(demo_dir, start_ts, cam_name="left")

        # read episode CSV once per sample (simple & practical). For speed, add a tiny cache (below).
        csv_path = os.path.join(demo_dir, "ee_hand.csv")
        df = pd.read_csv(csv_path)

        # --- overlay on RAW image (optional) ---
        if self.overlay:
            pts_r_world = self._fk_points_right_world(df.iloc[start_ts])
            img_bgr = self._project_draw_right_on_left(img_bgr, pts_r_world)

        # finally resize/normalize once
        image = self._resize_norm_rgb(img_bgr)

        end_ts = start_ts + self.chunk_size * self.stride
        # We built index so end_ts <= episode_len
        actions = [self._row_to_action(df.iloc[t]) for t in range(start_ts, end_ts, self.stride)]
        actions = np.stack(actions, axis=0)  # (chunk_size, 29)

        # Interleave zero token (left hand) with right-hand token, then pad to 32
        actions_split = np.empty((actions.shape[0] * 2, 29), dtype=np.float32)
        actions_split[0::2] = 0.0
        actions_split[1::2] = actions

        # need to pad from 29 - 32 dims
        actions_out = np.zeros((actions_split.shape[0], 32), dtype=np.float32)
        actions_out[:, :29] = actions_split

        state = actions_out[1].copy()  # first right-hand token as state

        return {
            "image":   image.astype(np.float32),   # (H, W, 3) in [0,1]
            "state":   state.astype(np.float32),   # (32,)
            "actions": actions_out.astype(np.float32),  # (2*chunk_size, 32)
            "task":    "place red cube into box",
        }

#################################################################
#################################################################
# UNCOMMENT ME to visualize the dataset!
#################################################################
#################################################################

# import os
# import numpy as np
# import cv2
# from torch.utils.data import DataLoader

# # ---- import your dataset class first ----
# # from your_file import GalaxeaDatasetKeypointsJoints

# def save_rgb01(path, rgb01):
#     """rgb01: (H,W,3) float32 in [0,1]"""
#     bgr_u8 = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]  # RGB->BGR
#     cv2.imwrite(path, bgr_u8)

# if __name__ == "__main__":
#     dataset_root = "/iris/projects/humanoid/tesollo_dataset/robot_data_0903/red_cube_inbox"  # change if needed

#     ds = GalaxeaDatasetKeypointsJoints(
#         dataset_dir=dataset_root,
#         chunk_size=8,
#         stride=3,
#         overlay=True,   # turn on drawing
#     )

#     dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

#     out_dir = "/iris/projects/humanoid/openpi/robot_vis"
#     os.makedirs(out_dir, exist_ok=True)

#     # grab and save first 8 samples
#     for i, batch in enumerate(dl):
#         img = batch["image"][0].numpy()          # (H,W,3) float32 [0,1]
#         state = batch["state"][0].numpy()        # (32,)
#         actions = batch["actions"][0].numpy()    # (2*chunk, 32)
#         task = batch["task"][0]

#         save_rgb01(os.path.join(out_dir, f"sample_{i:02d}.png"), img)
#         print(f"saved {os.path.join(out_dir, f'sample_{i:02d}.png')}  |  state {state.shape}  actions {actions.shape}  task={task}")

#         if i >= 7:
#             break

#     print(f"Done. Check images in: {os.path.abspath(out_dir)}")