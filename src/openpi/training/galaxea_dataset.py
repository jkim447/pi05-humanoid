
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

    def _project_draw_right_on_left(self, img_bgr, pts_map):
        h, w = img_bgr.shape[:2]
        fx, fy, cx, cy = K_LEFT[0,0], K_LEFT[1,1], K_LEFT[0,2], K_LEFT[1,2]

        # project
        proj = {}
        for name, Pw in pts_map.items():
            Pc = T_BASE_TO_CAM_LEFT @ np.array([Pw[0], Pw[1], Pw[2], 1.0], dtype=np.float64)
            z = Pc[2]
            if z <= 1e-6: 
                continue
            u = int(fx * Pc[0] / z + cx)
            v = int(fy * Pc[1] / z + cy)
            if 0 <= u < w and 0 <= v < h:
                proj[name] = (u, v, z)

        # draw points
        for name, (u, v, z) in proj.items():
            cv2.circle(img_bgr, (u, v), 4, (0, 255, 255), 7)  # cyan for right

        # draw lines (clipped to image)
        rect = (0, 0, w, h)
        # TODO: uncomment me for the line connections!
        for a, b in self.right_connections:
            if a in proj and b in proj:
                pt1, pt2 = proj[a][:2], proj[b][:2]
                ok, p1, p2 = cv2.clipLine(rect, pt1, pt2)
                if ok:
                    cv2.line(img_bgr, p1, p2, (200, 200, 0), 2)   # pale yellow
        return img_bgr


    def _load_img_raw_bgr(self, demo_dir, ts, cam_name="left"):
        p = os.path.join(demo_dir, cam_name, f"{ts:06d}.jpg")
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(p)
        return img  # raw BGR uint8

    def _resize_norm_rgb(self, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
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