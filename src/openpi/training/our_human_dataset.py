"""
uv run src/openpi/training/our_human_dataset.py
"""

import os
import cv2
import numpy as np
import torch
import pandas as pd
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from openpi.training.hand_keypoints_config import JOINT_COLOR_BGR, LEFT_FINGERS, RIGHT_FINGERS
import random

# MANO right-hand semantic names by index (0..20)
MANO_RIGHT_NAMES = [
    "rightHand",                         # 0 wrist
    "rightThumbKnuckle","rightThumbIntermediateBase","rightThumbIntermediateTip","rightThumbTip",  # 1..4
    "rightIndexFingerKnuckle","rightIndexFingerIntermediateBase","rightIndexFingerIntermediateTip","rightIndexFingerTip",  # 5..8
    "rightMiddleFingerKnuckle","rightMiddleFingerIntermediateBase","rightMiddleFingerIntermediateTip","rightMiddleFingerTip",  # 9..12
    "rightRingFingerKnuckle","rightRingFingerIntermediateBase","rightRingFingerIntermediateTip","rightRingFingerTip",  # 13..16
    "rightLittleFingerKnuckle","rightLittleFingerIntermediateBase","rightLittleFingerIntermediateTip","rightLittleFingerTip",  # 17..20
]

# --------- Camera intrinsics (raw resolution) ---------
K_LEFT = np.array([[730.2571411132812, 0.0, 637.2598876953125],
                   [0.0, 730.2571411132812, 346.41082763671875],
                   [0.0, 0.0, 1.0]], dtype=np.float64)

# --------- Extrinsics: cam->base given, we invert to base->cam ---------
# T_BASE_TO_CAM_LEFT = np.linalg.inv(np.array([
#     [ 0.00692993, -0.87310148,  0.48748926,  0.14062141],
#     [-0.99995006, -0.00956093, -0.00290894,  0.03612369],
#     [ 0.00720065, -0.48744476, -0.87312414,  0.46063114],
#     [ 0., 0., 0., 1. ]
# ], dtype=np.float64))

T_BASE_TO_CAM_LEFT = np.linalg.inv(np.array([
    [0.01988061, -0.43758429,  0.89895759,  0.14056752],
    [-0.9996933,   0.00457983,  0.02433772,  0.02539622],
    [-0.01476688, -0.89916573, -0.43735903,  0.43713101],
    [0.0, 0.0, 0.0, 1.0]
]))
R_B2C = T_BASE_TO_CAM_LEFT[:3, :3]

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
    """Global depth sort of edge segments + points (far→near)."""
    H, W = image_rgb_float.shape[:2]
    name_to_idx = {n:i for i,n in enumerate(names)}
    prims = []

    # Edges -> segments
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
            t0 = k / edge_segments
            t1 = (k+1) / edge_segments
            u0 = int(round(ua*(1-t0) + ub*t0)); v0 = int(round(va*(1-t0) + vb*t0))
            u1 = int(round(ua*(1-t1) + ub*t1)); v1 = int(round(va*(1-t1) + vb*t1))
            if not (0 <= u0 < W and 0 <= v0 < H and 0 <= u1 < W and 0 <= v1 < H):
                continue
            zmid = za*(1-(t0+t1)/2) + zb*((t0+t1)/2)
            col  = color_of.get(b, (210,210,210))
            prims.append(("edge", zmid, (u0,v0), (u1,v1), col))

    # Points
    for i, p in enumerate(uv):
        if p is None: continue
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
            cv2.circle(img_bgr, (u,v), pt_radius, col, -1, lineType=cv2.LINE_AA)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0


class HumanDatasetKeypointsJoints(Dataset):
    """
    Outputs:
      {
        "image":   (H, W, 3) float32 in [0,1], left camera frame, with wrist+5 tips overlaid,
        "state":   (44,) float32   -- absolute action at the start of the window (no deltas),
        "actions": (2*chunk_size, 44) float32
                   -- interleaved [zero, right, zero, right, ...],
                      where the right-hand sequence has translation deltas applied.
      }
    Notes:
      - Action layout matches your original: [pos_cam(3), ori6d(6), joints(20), tips(15)] = 44.
      - Only translation (first 3 dims) is made relative to the first frame in the window.
      - Overlay is drawn on the resized left image using a scaled K.
    """
    def __init__(self, dataset_dir: str, chunk_size: int, stride: int = 2,
                 img_height: int = 224, img_width: int = 224, overlay: bool = False,
                 custom_instruction: str = None,
                 calib_h: int = 720, calib_w: int = 1280):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.chunk_size  = chunk_size
        self.stride      = stride
        self.img_h       = img_height
        self.img_w       = img_width
        self.overlay     = overlay
        self.custom_instruction = custom_instruction
        self.calib_h     = calib_h
        self.calib_w     = calib_w

        # Column names expected in robot_commands.csv (same as your original)
        self.wrist_xyz  = ["right_wrist_x","right_wrist_y","right_wrist_z"]
        self.wrist_quat = ["right_wrist_qx","right_wrist_qy","right_wrist_qz","right_wrist_qw"]
        self.joint_cols = [f"right_hand_{i}" for i in range(20)]
        tip_ids = [4, 8, 12, 16, 20]  # MANO-style tip indices
        self.tip_cols = sum([[f"right_hand_kp{i}_x", f"right_hand_kp{i}_y", f"right_hand_kp{i}_z"] for i in tip_ids], [])
        self.kp_cols = sum([[f"right_hand_kp{i}_x", f"right_hand_kp{i}_y", f"right_hand_kp{i}_z"]
                            for i in range(21)], [])

        # Skeleton edges in MANO indexing (wrist to finger bases, each finger chain)
        self.mano_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),          # index
            (0, 9), (9,10), (10,11), (11,12),        # middle
            (0,13), (13,14), (14,15), (15,16),       # ring
            (0,17), (17,18), (18,19), (19,20),       # little
        ]


        # Gather episodes: supports folders like "demo_12" or "Demo12"
        def _demo_key(name: str):
            try:
                if name.startswith("demo_"): return int(name.split("demo_")[1])
                if name.startswith("Demo"):  return int(name.replace("Demo",""))
            except Exception:
                pass
            return 10**9

        episode_dirs = sorted(
            [os.path.join(self.dataset_dir, d) for d in os.listdir(self.dataset_dir)
             if os.path.isdir(os.path.join(self.dataset_dir, d)) and (d.startswith("demo_") or d.startswith("Demo"))],
            key=lambda p: _demo_key(os.path.basename(p))
        )

        # TODO: comment me out when training!
        # num_episodes_to_keep = 20
        # if len(episode_dirs) > num_episodes_to_keep:
        #     episode_dirs = random.sample(episode_dirs, num_episodes_to_keep)

        # Precompute valid episodes and lengths by reading robot_commands.csv
        self.episodes = []  # list of (demo_dir, length)
        for demo_dir in episode_dirs:
            csv_path = os.path.join(demo_dir, "robot_commands.csv")
            if not os.path.exists(csv_path):
                continue
            try:
                N = len(pd.read_csv(csv_path))
            except Exception:
                continue
            if N > 0:
                self.episodes.append((demo_dir, N))

        if not self.episodes:
            raise RuntimeError(f"No valid episodes under {self.dataset_dir}")

        # Build flat index of (ep_id, t0) requiring a full chunk window
        horizon = self.chunk_size * self.stride
        self.index = []
        # for ep_id, (_, N) in enumerate(self.episodes):
        #     last_start = N - horizon
        #     if last_start < 0:
        #         continue
        #     for t0 in range(0, last_start + 1, self.stride): # TODO: why the +1?
        #         self.index.append((ep_id, t0))

        for ep_id, (_, N) in enumerate(self.episodes):
            if N <= 0:
                continue
            # t0 = 0, stride, 2*stride, ..., up to N-1
            for t0 in range(0, N, self.stride):
                self.index.append((ep_id, t0))

    def __len__(self):
        return len(self.index)

    # --------- small helpers ---------
    @staticmethod
    def _to_hom(p3):
        return np.array([p3[0], p3[1], p3[2], 1.0], dtype=np.float64)

    def _pos_base_to_cam(self, pxyz):
        return (T_BASE_TO_CAM_LEFT @ self._to_hom(pxyz))[:3].astype(np.float32)

    def _rot_base_to_cam(self, R_base):
        return (R_B2C @ R_base).astype(np.float32)

    def _scaled_K_for_image(self, out_w: int, out_h: int) -> np.ndarray:
        sx = float(out_w) / float(self.calib_w)
        sy = float(out_h) / float(self.calib_h)
        K = K_LEFT.copy()
        K[0,0] *= sx; K[0,2] *= sx
        K[1,1] *= sy; K[1,2] *= sy
        return K


    @staticmethod
    def _project(K: np.ndarray, xyz_cam) -> tuple[int,int]:
        x, y, z = float(xyz_cam[0]), float(xyz_cam[1]), max(float(xyz_cam[2]), 1e-6)
        u = K[0,0] * (x / z) + K[0,2]
        v = K[1,1] * (y / z) + K[1,2]
        return int(round(u)), int(round(v))

    @staticmethod
    def _project_scaled(K_raw, xyz_cam, in_h, in_w, out_h, out_w):
        """
        Project a 3D point (camera frame) onto a resized image.
        We scale intrinsics by the resize factors.
        """
        sx = float(out_w) / float(in_w)
        sy = float(out_h) / float(in_h)
        K = K_raw.copy()
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy

        z = max(float(xyz_cam[2]), 1e-6)
        u = K[0, 0] * (float(xyz_cam[0]) / z) + K[0, 2]
        v = K[1, 1] * (float(xyz_cam[1]) / z) + K[1, 2]
        return int(round(u)), int(round(v))

    @staticmethod
    def _safe_imread(path):
        im = cv2.imread(path)
        return im

    def _row_keypoints_cam(self, row) -> np.ndarray:
        """
        Return all 21 MANO keypoints transformed to the *camera* frame.
        Shape: (21, 3), dtype float32.
        """
        pts = []
        for i in range(0, len(self.kp_cols), 3):
            px = row[self.kp_cols[i+0]]
            py = row[self.kp_cols[i+1]]
            pz = row[self.kp_cols[i+2]]
            pts.append(self._pos_base_to_cam([px, py, pz]))
        return np.stack(pts, axis=0).astype(np.float32)  # (21, 3)

    def _draw_hand_skeleton(self, img_bgr: np.ndarray, kps_cam: np.ndarray,
                        raw_h: int, raw_w: int, out_h: int, out_w: int) -> np.ndarray:
        # We draw on the raw image (raw_h, raw_w), so scale K to that size once.
        K_img = self._scaled_K_for_image(raw_w, raw_h)

        names = MANO_RIGHT_NAMES
        uv, z = [], []
        for (x, y, zc) in kps_cam:
            u, v = self._project(K_img, (x, y, zc))
            uv.append((u, v))
            z.append(zc)
        z = np.array(z, dtype=np.float32)

        edges_by_name = [(names[i], names[j]) for (i, j) in self.mano_edges if i < len(names) and j < len(names)]
        fallback = (210, 210, 210)
        color_of = {n: JOINT_COLOR_BGR.get(n, fallback) for n in names}

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        img_rgb = draw_skeleton_occlusion_aware(
            img_rgb, names=names, uv=uv, z=z,
            edges_by_name=edges_by_name, color_of=color_of,
            pt_radius=4, line_thickness=8, edge_segments=12, # TODO: align this with the robot
        )
        return cv2.cvtColor((img_rgb*255).astype(np.uint8), cv2.COLOR_RGB2BGR) # BGR


    def _row_to_action(self, row):
        # position in camera frame
        p_cam = self._pos_base_to_cam([row[c] for c in self.wrist_xyz])  # (3,)

        # orientation 6D in camera frame
        R_base = R.from_quat([row[c] for c in self.wrist_quat]).as_matrix()
        R_cam  = self._rot_base_to_cam(R_base)
        ori6d  = R_cam[:, :2].reshape(-1, order="F").astype(np.float32)  # (6,)

        # hand joints
        joints = np.asarray([row[c] for c in self.joint_cols], dtype=np.float32)  # (20,)

        # tip points to camera frame
        tips_cam = []
        for i in range(0, len(self.tip_cols), 3):
            tips_cam.append(self._pos_base_to_cam([
                row[self.tip_cols[i]],
                row[self.tip_cols[i+1]],
                row[self.tip_cols[i+2]],
            ]))
        tips_cam = np.concatenate(tips_cam, axis=0).astype(np.float32)  # (15,)

        # final 44D action
        # return np.concatenate([p_cam, ori6d, joints, tips_cam], axis=0).astype(np.float32)
        return np.concatenate([p_cam, ori6d, tips_cam], axis=0).astype(np.float32) #: add joints back in later when you get it


    def _load_left_image(self, demo_dir, t):
        path = os.path.join(demo_dir, "left", f"{t:06d}.jpg")
        bgr  = self._safe_imread(path)
        if bgr is None:
            raise FileNotFoundError(f"Missing image: {path}")
        h0, w0 = bgr.shape[:2]
        return bgr, (h0, w0)   # return RAW BGR, no resize, keep original size

    def _row_wrist_cam_pos6(self, row):
        # position in camera frame
        p_cam = self._pos_base_to_cam([row[c] for c in self.wrist_xyz])  # (3,)
        # orientation 6D in camera frame
        R_base = R.from_quat([row[c] for c in self.wrist_quat]).as_matrix()
        R_cam  = self._rot_base_to_cam(R_base)
        rot6   = R_cam[:, :2].reshape(-1, order="F").astype(np.float32)  # (6,)
        return p_cam.astype(np.float32), rot6

    def _row_joints20(self, row):
        return np.asarray([row[c] for c in self.joint_cols], dtype=np.float32)  # (20,)

    @staticmethod
    def _to_rgb_uint8_and_resize(bgr_img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        # TODO: MAKE SURE THIS IS ACTIVE!
        rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return rgb.astype(np.uint8)


    def __getitem__(self, index):
        ep_id, t0 = self.index[index]
        demo_dir, N = self.episodes[ep_id]

        # image (raw)
        try:
            bgr_raw, (raw_h, raw_w) = self._load_left_image(demo_dir, t0)
        except FileNotFoundError:
            new_index = np.random.randint(0, len(self))
            return self.__getitem__(new_index)

        # csv
        csv_path = os.path.join(demo_dir, "robot_commands.csv")
        df = pd.read_csv(csv_path)

        # --- state (32): [L pos3+rot6 zeros(9), R pos3+rot6 abs(9), L joints[1:8] zeros(7), R joints[1:8] abs(7)] ---
        row0 = df.iloc[t0]
        R_pos0, R_rot60 = self._row_wrist_cam_pos6(row0)
        qR0 = self._row_joints20(row0)

        L_posrot6_zeros = np.zeros(9, np.float32)
        L_j7_zeros      = np.zeros(7, np.float32)
        R_j7_abs        = qR0[1:8].astype(np.float32)  # right_hand_1 .. right_hand_7

        state = np.concatenate([L_posrot6_zeros, R_pos0, R_rot60, L_j7_zeros, R_j7_abs], dtype=np.float32)

        # --- actions (2*chunk, 29): interleaved [zeros(29), right(Δpos3, Δrot6, Δjoints20)] ---
        # reference at t0
        ref_pos, ref_rot6, ref_qR = R_pos0, R_rot60, qR0

        actions_lr = []
        for dt in range(self.chunk_size):
            t = t0 + dt * self.stride
            if t < N:
                row_t = df.iloc[t]
                pos_t, rot6_t = self._row_wrist_cam_pos6(row_t)
                qR_t = self._row_joints20(row_t)

                dpos  = (pos_t  - ref_pos).astype(np.float32)   # (3,)
                drot6 = (rot6_t - ref_rot6).astype(np.float32)  # (6,)
                dqR   = (qR_t   - ref_qR).astype(np.float32)    # (20,)

                right = np.concatenate([dpos, drot6, dqR], axis=0).astype(np.float32)  # (29,)
                actions_lr.extend([np.zeros(29, np.float32), right])
            else:
                # out of range → pad both tokens with zeros (explicit and easy to read)
                actions_lr.extend([np.zeros(29, np.float32), np.zeros(29, np.float32)])

        actions_out = np.stack(actions_lr, axis=0).astype(np.float32)  # (2*chunk, 29)

        # --- visualization (optional) & final image as uint8 ---
        if self.overlay:
            vis_raw = bgr_raw.copy()
            kps_cam = self._row_keypoints_cam(df.iloc[t0])  # (21,3) in cam
            vis_raw = self._draw_hand_skeleton(vis_raw, kps_cam, raw_h, raw_w, raw_h, raw_w) # BGR
            img = self._to_rgb_uint8_and_resize(vis_raw, self.img_w, self.img_h)
        else:
            img = self._to_rgb_uint8_and_resize(bgr_raw, self.img_w, self.img_h)


        H, W, C = img.shape
        zeros_img = np.zeros_like(img, dtype=np.uint8)

        return {
            "image":   img.astype(np.uint8),      # (H,W,3) uint8 RGB
            "wrist_image_left":  zeros_img,                    # dummy
            "wrist_image_right": zeros_img,                    # dummy
            "state":   state.astype(np.float32),  # (32,)
            "actions": actions_out.astype(np.float32),  # (2*chunk, 29)
            "task":    self.custom_instruction if self.custom_instruction is not None else "vertical_pick_place",
    }


# test_human_dataset_simple.py
# import os
# import cv2
# import torch
# from torch.utils.data import DataLoader

# # import the dataset class from wherever you saved it
# # from my_datasets import HumanDatasetKeypointsJoints_Simple
# # from human_dataset_simple import HumanDatasetKeypointsJoints_Simple  # <-- adjust path/name

# def _ensure_dir(d):
#     if not os.path.exists(d):
#         os.makedirs(d, exist_ok=True)

# def save_rgb_image(rgb_tensor_or_array, out_path):
#     """
#     rgb_tensor_or_array: (H, W, 3) in [0,1], torch.Tensor or np.ndarray
#     Saves as JPG using OpenCV (converts RGB->BGR).
#     """
#     if isinstance(rgb_tensor_or_array, torch.Tensor):
#         img = rgb_tensor_or_array.detach().cpu().numpy()
#     else:
#         img = rgb_tensor_or_array
#     # img = (img.clip(0, 1) * 255.0).astype("uint8")
#     bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(out_path, bgr)

# def main():
#     ds = HumanDatasetKeypointsJoints(
#         # dataset_dir="/iris/projects/humanoid/hamer/keypoint_human_data_red_inbox",
#         dataset_dir = "/iris/projects/humanoid/dataset/HUMAN_SORT_TL_1101",
#         chunk_size=20,
#         stride=2,
#         img_height=224,
#         img_width=224,
#         overlay=True,   # draws wrist + 5 tips on the resized left image
#         custom_instruction="vertical_pick_place",
#     )

#     loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

#     out_dir = "vis_samples_simple"
#     _ensure_dir(out_dir)

#     for i, batch in enumerate(loader):
#         # batch is a dict with:
#         #   image:   (B, H, W, 3) float32 in [0,1]
#         #   state:   (B, 44)
#         #   actions: (B, 2*chunk_size, 44)
#         #   task:    list[str] of length B
#         image   = batch["image"][0]      # (H, W, 3)
#         state   = batch["state"][0]      # (44,)
#         actions = batch["actions"][0]    # (2*chunk, 44)

#         out_img = os.path.join(out_dir, f"sample_{i:02d}.jpg")
#         save_rgb_image(image, out_img)

#         # quick shape + small value checks
#         print(f"[{i}] saved:", out_img)
#         print("   state:",   tuple(state.shape))
#         print("   actions:", tuple(actions.shape))
#         # show first few dims for a quick glance
#         with torch.no_grad():
#             s_np = state.detach().cpu().numpy()
#             a_np = actions.detach().cpu().numpy()
#         print("   state[:8]:", s_np[:8])
#         print("   actions[1, :8] (first right-hand token):", a_np[1, :8])

#         if i >= 10:  # first 5 samples only
#             break

# if __name__ == "__main__":
#     main()
