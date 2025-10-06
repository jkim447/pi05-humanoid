
import os
import cv2
import numpy as np
import torch
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

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
    def __init__(self, dataset_dir, chunk_size, stride = 3):
        super(GalaxeaDatasetKeypointsJoints).__init__()
        self.dataset_dir = dataset_dir
        self.chunk_size = chunk_size
        self.img_height = 224
        self.img_width  = 224
        self.stride = stride
      
        self.right_hand_cols = [f"right_hand_{i}" for i in range(20)]
        self.right_actual_hand_cols = [f"right_actual_hand_{i}" for i in range(20)]
        self.action_camera = "left"  # which camera to use for action ref frame
        
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
            print("going thru:", ep_id)
            last_start = N - horizon
            if last_start < 0:
                continue
            # t0 = 0, stride, 2*stride, ..., <= last_start
            self.index.extend((ep_id, t0) for t0 in range(0, last_start + 1, self.stride))

    def __len__(self):
        return len(self.index)

    def load_img(self, demo_dir, start_ts, cam_name = "left"):
        img_path = os.path.join(demo_dir, cam_name, f"{start_ts:06d}.jpg")
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img_rgb  = cv2.cvtColor(img_bgr,  cv2.COLOR_BGR2RGB)
        img_rgb  = cv2.resize(img_rgb,  (self.img_width, self.img_height))
        img_rgb  = img_rgb.astype(np.float32) / 255.0  # normalize to 0–1
        return img_rgb

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
        image = self.load_img(demo_dir, start_ts, cam_name="left")

        # read episode CSV once per sample (simple & practical). For speed, add a tiny cache (below).
        csv_path = os.path.join(demo_dir, "ee_hand.csv")
        df = pd.read_csv(csv_path)

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

# def save_images(images, out_dir, idx):
#     """Save (2,C,H,W) tensor images as jpg."""
#     os.makedirs(out_dir, exist_ok=True)
#     imgs = images.permute(0,2,3,1).cpu().numpy()  # -> (2,H,W,C), float [0,1]
#     for cam_id in range(imgs.shape[0]):
#         img = (imgs[cam_id] * 255).astype("uint8")[:, :, ::-1]  # RGB->BGR for cv2
#         out_path = os.path.join(out_dir, f"sample{idx}_cam{cam_id}.jpg")
#         cv2.imwrite(out_path, img)
#         print(f"Saved {out_path}")


# # --- minimal fingertip overlay (left cam) ---
# ORIG_W, ORIG_H = 1280, 720  # set to your raw capture size
# Sx, Sy = 224/ORIG_W, 224/ORIG_H
# K224 = np.array([[K_LEFT[0,0]*Sx, 0, K_LEFT[0,2]*Sx],
#                  [0, K_LEFT[1,1]*Sy, K_LEFT[1,2]*Sy],
#                  [0, 0, 1]], dtype=np.float32)

# # optionally use this
# def save_with_tips_fullres(images_2chw, actions_ta, out_dir, idx):
#     os.makedirs(out_dir, exist_ok=True)
#     # images_2chw: (2,C,H,W), already full-res RGB in [0,1]
#     imgs = (images_2chw.permute(0,2,3,1).cpu().numpy() * 255).astype("uint8")  # (2,H,W,C)

#     # Save both cams first
#     for cam in range(2):
#         out_path = os.path.join(out_dir, f"sample{idx}_cam{cam}.jpg")
#         cv2.imwrite(out_path, imgs[cam][:,:,::-1])  # RGB->BGR

#     # Get fingertip positions from action (left cam frame, 3D)
#     a0 = actions_ta[0].cpu().numpy()
#     tips = a0[-15:].reshape(5,3).astype(np.float32)  # (Xc,Yc,Zc)

#     # Project with original intrinsics K_LEFT
#     Z = np.clip(tips[:,2:3], 1e-6, None)
#     xn = tips[:,:2] / Z                     # (x/z, y/z)
#     uv = (K_LEFT[:2,:2] @ xn.T).T + K_LEFT[:2,2]  # (5,2) pixels in full res

#     # Draw dots on left cam image
#     left_path = os.path.join(out_dir, f"sample{idx}_cam0.jpg")
#     im = cv2.imread(left_path)  # full-res BGR
#     for u,v in uv:
#         u,v = int(round(u)), int(round(v))
#         if 0 <= u < im.shape[1] and 0 <= v < im.shape[0]:
#             cv2.circle(im, (u,v), 8, (0,0,255), -1)  # red dots
#     cv2.imwrite(os.path.join(out_dir, f"sample{idx}_cam0_with_tips.jpg"), im)

# def save_with_tips(images_2chw, actions_ta, out_dir, idx):
#     os.makedirs(out_dir, exist_ok=True)
#     imgs = (images_2chw.permute(0,2,3,1).cpu().numpy() * 255).astype("uint8")  # (2,H,W,C) RGB
#     # Save both cams first
#     for cam in range(2):
#         cv2.imwrite(os.path.join(out_dir, f"sample{idx}_cam{cam}.jpg"), imgs[cam][:,:,::-1])

#     # Get first timestep action and its last 15 dims -> (5,3) cam-frame points
#     a0 = actions_ta[0].cpu().numpy()
#     tips = a0[-15:].reshape(5,3).astype(np.float32)  # (Xc,Yc,Zc)
#     Z = np.clip(tips[:,2:3], 1e-6, None)
#     xn = tips[:,:2] / Z                               # (x/z, y/z)
#     uv = (K224[:2,:2] @ xn.T).T + K224[:2,2]          # (5,2) in 224x224

#     # Draw on left image (cam0) and resave
#     left_path = os.path.join(out_dir, f"sample{idx}_cam0.jpg")
#     im = cv2.imread(left_path)                        # BGR 224x224
#     for u,v in uv:
#         u,v = int(round(u)), int(round(v))
#         if 0 <= u < im.shape[1] and 0 <= v < im.shape[0]:
#             cv2.circle(im, (u,v), 5, (255,0,255), -1)  # red dots
#     cv2.imwrite(os.path.join(out_dir, f"sample{idx}_cam0_with_tips.jpg"), im)


# ds = GalaxeaDatasetKeypointsJoints(
#     dataset_dir="/iris/projects/humanoid/tesollo_dataset/robot_data_0903/red_cube_inbox",
#     chunk_size=20,
#     apply_data_aug=True,
#     normalize=False,
#     compute_keypoints=True,
#     overlay_keypoints=False   # skeletons drawn before resizing
# )

# loader = DataLoader(ds, batch_size=1, shuffle=True)

# out_dir = "vis_samples"
# for i, batch in enumerate(loader):
#     image_data, qpos, action, is_pad = batch
#     save_images(image_data[0], out_dir, i)
#     # save_with_tips(image_data[0], action[0], out_dir, i)
#     save_with_tips_fullres(image_data[0], action[0], out_dir, i)
#     print("qpos:", qpos.shape, "action:", action.shape, "is_pad:", is_pad.shape)
#     if i >= 4:   # save first 5 samples only
#         break

# # (optional) compare with augmentation OFF
# # ds_noaug = GalaxeaDataset(dataset_dir="/path/to/Galaxea", chunk_size=50, apply_data_aug=False, normalize=True)
# # dump_dataset_images(ds_noaug, out_dir="viz_out", num_samples=20, prefix="orig")
