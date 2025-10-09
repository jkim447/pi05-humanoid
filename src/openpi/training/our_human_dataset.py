import os
import cv2
import numpy as np
import torch
import pandas as pd
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

# --------- Camera intrinsics (raw resolution) ---------
K_LEFT = np.array([[730.2571411132812, 0.0, 637.2598876953125],
                   [0.0, 730.2571411132812, 346.41082763671875],
                   [0.0, 0.0, 1.0]], dtype=np.float64)

# --------- Extrinsics: cam->base given, we invert to base->cam ---------
T_BASE_TO_CAM_LEFT = np.linalg.inv(np.array([
    [ 0.00692993, -0.87310148,  0.48748926,  0.14062141],
    [-0.99995006, -0.00956093, -0.00290894,  0.03612369],
    [ 0.00720065, -0.48744476, -0.87312414,  0.46063114],
    [ 0., 0., 0., 1. ]
], dtype=np.float64))
R_B2C = T_BASE_TO_CAM_LEFT[:3, :3]

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
                 custom_instruction: str = None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.chunk_size  = chunk_size
        self.stride      = stride
        self.img_h       = img_height
        self.img_w       = img_width
        self.overlay     = overlay
        self.custom_instruction = custom_instruction

        # Column names expected in robot_commands.csv (same as your original)
        self.wrist_xyz  = ["right_wrist_x","right_wrist_y","right_wrist_z"]
        self.wrist_quat = ["right_wrist_qx","right_wrist_qy","right_wrist_qz","right_wrist_qw"]
        self.joint_cols = [f"right_hand_{i}" for i in range(20)]
        tip_ids = [4, 8, 12, 16, 20]  # MANO-style tip indices
        self.tip_cols = sum([[f"right_hand_kp{i}_x", f"right_hand_kp{i}_y", f"right_hand_kp{i}_z"] for i in tip_ids], [])

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
        for ep_id, (_, N) in enumerate(self.episodes):
            last_start = N - horizon
            if last_start < 0:
                continue
            for t0 in range(0, last_start + 1, self.stride): # TODO: why the +1?
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
        return np.concatenate([p_cam, ori6d, tips_cam], axis=0).astype(np.float32) # TODO: add joints back in later when you get it


    def _load_left_image(self, demo_dir, t):
        path = os.path.join(demo_dir, "left", f"{t:06d}.jpg")
        bgr  = self._safe_imread(path)
        if bgr is None:
            raise FileNotFoundError(f"Missing image: {path}")
        h0, w0 = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_w, self.img_h))
        rgb = (rgb.astype(np.float32))
        return rgb, (h0, w0)

    def __getitem__(self, index):
        ep_id, t0 = self.index[index]
        demo_dir, N = self.episodes[ep_id]

        # load start image and remember original size for projection scaling
        img, (raw_h, raw_w) = self._load_left_image(demo_dir, t0)
        img = img/255.0

        # load CSV
        csv_path = os.path.join(demo_dir, "robot_commands.csv")
        df = pd.read_csv(csv_path)

        # build action sequence
        end_t = t0 + self.chunk_size * self.stride
        seq = [self._row_to_action(df.iloc[t]) for t in range(t0, end_t, self.stride)]
        actions = np.stack(seq, axis=0).astype(np.float32)  # (chunk, 44)

        # keep absolute at first step for state
        # TODO: be careful, since if you have single arm data the first item will be the left arm by default
        # but if you have dual arm data it will be the right arm.
        state_abs = actions[0].copy()  # (44,)

        # deltas only on translation across the window (first 3 components)
        # actions[:, 0:3] -= actions[0, 0:3]

        # interleave zero (left hand placeholder) and right-hand token
        zeros = np.zeros((actions.shape[0], actions.shape[1]), dtype=np.float32)
        actions_out = np.empty((actions.shape[0] * 2, actions.shape[1]), dtype=np.float32)
        actions_out[0::2] = zeros
        actions_out[1::2] = actions

        # draw overlay on the resized left image using scaled intrinsics and absolute state
        if self.overlay:
            vis = (img * 255.0).astype(np.uint8).copy()
            # wrist
            wrist_cam = state_abs[:3]
            u, v = self._project_scaled(K_LEFT, wrist_cam, raw_h, raw_w, self.img_h, self.img_w)
            if 0 <= u < self.img_w and 0 <= v < self.img_h:
                cv2.circle(vis, (u, v), 7, (0, 255, 0), -1)  # green

            # tips
            tips_flat = state_abs[-15:].reshape(5, 3)
            for x, y, z in tips_flat:
                u, v = self._project_scaled(K_LEFT, (x, y, z), raw_h, raw_w, self.img_h, self.img_w)
                if 0 <= u < self.img_w and 0 <= v < self.img_h:
                    cv2.circle(vis, (u, v), 6, (255, 0, 0), -1)  # red
            img = vis.astype(np.float32) / 255.0

        return {
            "image":   img.astype(np.float32),      # (H, W, 3) in [0,1]
            "state":   state_abs.astype(np.float32),# (44,) absolute
            "actions": actions_out.astype(np.float32),  # (2*chunk, 44) with translation deltas
            "task": self.custom_instruction if self.custom_instruction is not None else "place red cube into box",

        }


# test_human_dataset_simple.py
import os
import cv2
import torch
from torch.utils.data import DataLoader

# import the dataset class from wherever you saved it
# from my_datasets import HumanDatasetKeypointsJoints_Simple
# from human_dataset_simple import HumanDatasetKeypointsJoints_Simple  # <-- adjust path/name

def _ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_rgb_image(rgb_tensor_or_array, out_path):
    """
    rgb_tensor_or_array: (H, W, 3) in [0,1], torch.Tensor or np.ndarray
    Saves as JPG using OpenCV (converts RGB->BGR).
    """
    if isinstance(rgb_tensor_or_array, torch.Tensor):
        img = rgb_tensor_or_array.detach().cpu().numpy()
    else:
        img = rgb_tensor_or_array
    img = (img.clip(0, 1) * 255.0).astype("uint8")
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)

def main():
    ds = HumanDatasetKeypointsJoints(
        dataset_dir="/iris/projects/humanoid/hamer/keypoint_human_data_red_inbox",
        chunk_size=20,
        stride=2,
        img_height=224,
        img_width=224,
        overlay=False,   # draws wrist + 5 tips on the resized left image
    )

    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    out_dir = "vis_samples_simple"
    _ensure_dir(out_dir)

    for i, batch in enumerate(loader):
        # batch is a dict with:
        #   image:   (B, H, W, 3) float32 in [0,1]
        #   state:   (B, 44)
        #   actions: (B, 2*chunk_size, 44)
        #   task:    list[str] of length B
        image   = batch["image"][0]      # (H, W, 3)
        state   = batch["state"][0]      # (44,)
        actions = batch["actions"][0]    # (2*chunk, 44)

        out_img = os.path.join(out_dir, f"sample_{i:02d}.jpg")
        save_rgb_image(image, out_img)

        # quick shape + small value checks
        print(f"[{i}] saved:", out_img)
        print("   state:",   tuple(state.shape))
        print("   actions:", tuple(actions.shape))
        # show first few dims for a quick glance
        with torch.no_grad():
            s_np = state.detach().cpu().numpy()
            a_np = actions.detach().cpu().numpy()
        print("   state[:8]:", s_np[:8])
        print("   actions[1, :8] (first right-hand token):", a_np[1, :8])

        if i >= 4:  # first 5 samples only
            break

if __name__ == "__main__":
    main()
