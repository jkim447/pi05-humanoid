"""
Usage (for testing):
uv run src/openpi/training/egodex_dataset.py
"""
import os, glob, cv2, h5py
import numpy as np
from typing import List, Tuple, Dict, Any, Literal, Optional
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
import json
from dataclasses import asdict, dataclass
import random

# ---------- small SE(3) helpers ----------
def _inv(T: np.ndarray) -> np.ndarray:
    Rm = T[:3, :3]
    t  = T[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = Rm.T
    inv[:3, 3]  = -Rm.T @ t
    return inv

def _rotmat_to_rot6d(Rm: np.ndarray) -> np.ndarray:
    # [r00, r10, r20, r01, r11, r21]
    return np.array([Rm[0,0], Rm[1,0], Rm[2,0], Rm[0,1], Rm[1,1], Rm[2,1]], dtype=np.float32)

def _pose_world_to_cam(T_world_obj: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    return _inv(T_world_cam) @ T_world_obj


@dataclass
class Episode:
    h5: str
    mp4: str
    N: int

def _index_path(root_dir: str) -> str:
    return os.path.join(root_dir, "episodes_index.jsonl")

def _save_index(root_dir: str, episodes: list[Episode]) -> None:
    path = _index_path(root_dir)
    with open(path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(asdict(ep)) + "\n")

def _load_index(root_dir: str) -> list[Episode] | None:
    path = _index_path(root_dir)
    if not os.path.exists(path):
        return None
    out: list[Episode] = []
    with open(path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                # cheap validation: files still exist
                if os.path.exists(obj["h5"]) and os.path.exists(obj["mp4"]):
                    out.append(Episode(h5=obj["h5"], mp4=obj["mp4"], N=int(obj["N"])))
            except Exception:
                continue
    return out

# ---------- simplified, windowed dataset ----------
class EgoDexSeqDataset(Dataset):
    """
    Map-style dataset that yields *windows*:
      sample = {
        "image":   (H, W, 3) uint8 RGB at start frame (resized to image_size),
        "state":   (D,) float32  — state at start frame,
        "actions": (H, D) float32 — sequence of H = action_horizon states,
        "task":    str (from HDF5 attrs['llm_description'] if present)
      }

    state_format:
      - "pi0": [L_xyz(3), L_rot6d(6), 0.0, R_xyz(3), R_rot6d(6), 0.0, zeros(12)] -> 32-D
      - "ego": [L (xyz+quat 7), R (xyz+quat 7), 10 fingertips xyz (30)] -> 44-D
    """
    def __init__(
        self,
        root_dir: str,
        action_horizon: int,
        image_size: tuple[int, int] = (224, 224),
        state_format: Literal["pi0", "ego", "ego_split"] = "ego", # ego_split means split left/right hand across two tokens for both actions and states
        window_stride: int = 1,                 # step for consecutive windows
        traj_per_task: Optional[int] = None,    # optional cap per task
        max_episodes: Optional[int] = None,     # optional global cap
        rebuild_index: bool = False,   # <— NEW
        load_images: bool = True,
    ):
        assert action_horizon >= 1
        self.root_dir = root_dir
        self.image_size = image_size
        self.state_format = state_format
        self.H = int(action_horizon)
        self.stride = max(1, int(window_stride))
        self.load_images = load_images

        self.wrists = ["leftHand", "rightHand"]
        self.fingertips = [
            "leftThumbTip","leftIndexFingerTip","leftMiddleFingerTip","leftRingFingerTip","leftLittleFingerTip",
            "rightThumbTip","rightIndexFingerTip","rightMiddleFingerTip","rightRingFingerTip","rightLittleFingerTip",
        ]

        # 1) collect (h5, mp4, N) per episode, with on-disk cache
        episodes: list[Episode] | None = None
        if not rebuild_index:
            episodes = _load_index(self.root_dir)
            print("successfully loaded cached index:", len(episodes) if episodes else 0, "episodes")

            # ---- pick a fixed number of episodes per task ----
            # TODO: change as needed (ideally delete for training)
            episodes_per_task = 50        # <= choose how many per task
            rng = random.Random(42)      # <= set seed for reproducibility (or remove)

            def _task_name(ep: Episode) -> str:
                # task is the folder right above the file, e.g.
                # .../part3/open_close_insert_remove_case/811.hdf5  -> "open_close_insert_remove_case"
                return os.path.basename(os.path.dirname(ep.h5))

            # group by task
            by_task: dict[str, list[Episode]] = {}
            for ep in episodes:
                by_task.setdefault(_task_name(ep), []).append(ep)

            # sample K per task (or all if fewer)
            picked: list[Episode] = []
            for task, eps in by_task.items():
                rng.shuffle(eps)                 # simple random; remove if you prefer sorted order
                picked.extend(eps[:episodes_per_task])

            # optional: shuffle overall, then cap with max_episodes if you also want a global limit
            # rng.shuffle(picked)
            # if (max_episodes is not None) and (len(picked) > max_episodes):
            #     picked = picked[:max_episodes]

            episodes = picked

        if episodes is None:
            print("scanning dataset for (hdf5, mp4) pairs, this may take a while...")
            episodes = []
            # TODO: uncomment me!
            part_dirs = sorted(
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d)) and (d.startswith("part") or d in ("test","extra"))
            )

            # TODO: comment me out!
            # part_dirs = ["/iris/projects/humanoid/dataset/ego_dex/part1"]
            print("found part dirs:", part_dirs)

            for part in part_dirs:
                part_path = os.path.join(root_dir, part)
                for task in sorted(os.listdir(part_path)):
                    print("I'm scanning through:", part_path, task)
                    task_path = os.path.join(part_path, task)
                    if not os.path.isdir(task_path):
                        print("ALERT! Skipping non-directory:", task_path)
                        continue

                    h5_files = sorted(glob.glob(os.path.join(task_path, "*.hdf5")))
                    pairs = []
                    for h5f in h5_files:
                        mp4f = h5f.replace(".hdf5", ".mp4")
                        if os.path.exists(mp4f):
                            pairs.append((h5f, mp4f))

                    # TODO: uncomment if you want to limit to few trajs since it takes forever, but possibly irrelevant at this point since
                    # I've scoured through the entire dataset and saved all the episodes to a file to read from later
                    # traj_per_task = 5
                    if traj_per_task is not None and len(pairs) > traj_per_task:
                        idxs = np.random.choice(len(pairs), size=traj_per_task, replace=False)
                        pairs = [pairs[i] for i in idxs]

                    for h5f, mp4f in pairs:
                        try:
                            with h5py.File(h5f, "r") as f:
                                N = int(f["transforms"]["leftHand"].shape[0])
                            if N >= self.H:
                                print(h5f, mp4)
                                episodes.append(Episode(h5=h5f, mp4=mp4f, N=N))
                        except Exception:
                            continue

            # persist for future runs
            _save_index(self.root_dir, episodes)
            
        # keep tuple form used later
        self.episodes = [(ep.h5, ep.mp4, ep.N) for ep in episodes]

        index = []
        for ep_id, (_, _, N) in enumerate(self.episodes):
            last_start = N - self.H  # TODO: this is not good, need to pad!
            index.extend((ep_id, t0) for t0 in range(0, last_start + 1, self.stride))
        self.index: List[Tuple[int, int]] = index

        # assert False

    def __len__(self) -> int:
        return len(self.index)

    # ---- state builders ----
    def _state_pi0(self, f: h5py.File, t: int) -> np.ndarray:
        T_world_cam = f["transforms"]["camera"][t]
        L_world = f["transforms"]["leftHand"][t]
        R_world = f["transforms"]["rightHand"][t]
        L_cam = _pose_world_to_cam(L_world, T_world_cam)
        R_cam = _pose_world_to_cam(R_world, T_world_cam)
        L_pos = L_cam[:3, 3].astype(np.float32)
        R_pos = R_cam[:3, 3].astype(np.float32)
        L_rot6 = _rotmat_to_rot6d(L_cam[:3, :3])
        R_rot6 = _rotmat_to_rot6d(R_cam[:3, :3])
        hands_pad = np.zeros((12,), dtype=np.float32)
        return np.concatenate([L_pos, L_rot6, [0.0], R_pos, R_rot6, [0.0], hands_pad], dtype=np.float32)

    def _state_ego(self, f: h5py.File, t: int) -> np.ndarray:
        T_world_cam = f["transforms"]["camera"][t]
        # wrists xyz+quat
        ws = []
        for joint in self.wrists:
            T_world = f["transforms"][joint][t]
            T_cam   = _pose_world_to_cam(T_world, T_world_cam)
            pos  = T_cam[:3, 3].astype(np.float32)
            rot6 = _rotmat_to_rot6d(T_cam[:3, :3])             # (6,) from first two R columns
            ws.append(np.concatenate([pos, rot6], dtype=np.float32))
        wrists_vec = np.concatenate(ws, dtype=np.float32)  # 18
        # fingertips xyz
        tips = []
        for joint in self.fingertips:
            T_world = f["transforms"][joint][t]
            T_cam   = _pose_world_to_cam(T_world, T_world_cam)
            tips.append(T_cam[:3, 3].astype(np.float32))
        tips_vec = np.concatenate(tips, dtype=np.float32)  # 30
        return np.concatenate([wrists_vec, tips_vec], dtype=np.float32)  # 48

    def _state_vec(self, f: h5py.File, t: int) -> np.ndarray:
        return self._state_ego(f, t) if self.state_format in ("ego", "ego_split") else self._state_pi0(f, t)

    # ---- IO helpers ----
    def _read_rgb(self, mp4_path: str, t: int) -> np.ndarray: 
        H, W = self.image_size

        def _blank() -> np.ndarray:
            return np.zeros((H, W, 3), dtype=np.float32)

        # Try a few times (helps with flaky decoders)
        for attempt in range(3):
            cap = cv2.VideoCapture(mp4_path)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
            ok, frame_bgr = cap.read()
            cap.release()

            if ok and frame_bgr is not None:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
                return rgb.astype(np.float32) / 255.0

        # Fallback: use blank image if all attempts fail
        print(f"[WARN] Failed to read frame {t} from {mp4_path}. Using blank image.")
        return _blank()

    # ---- main fetch ----
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_id, t0 = self.index[idx]
        h5_path, mp4_path, N = self.episodes[ep_id]

        # image at start frame (you can switch to center frame if you prefer)
        # image at start frame (you can switch to center frame if you prefer)
        if self.load_images:
            image = self._read_rgb(mp4_path, t0)
        else:
            H, W = self.image_size
            image = np.zeros((H, W, 3), dtype=np.float32)  # <— NEW: no image I/O for stats

        with h5py.File(h5_path, "r") as f:

            if self.state_format == "ego_split":
                actions = np.stack([self._state_vec(f, t0 + dt) for dt in range(self.H)], axis=0)  # (H, D)
                D = actions.shape[1]
                if D != 48:
                    raise ValueError(f"ego_split expects 48-D state before split, got {D}")

                # split into two 24-D tokens
                first_half  = actions[:, :24]
                second_half = actions[:, 24:]
                actions_split = np.empty((actions.shape[0] * 2, 24), dtype=np.float32)
                actions_split[0::2] = first_half
                actions_split[1::2] = second_half

                # pad each token to 32-D
                actions_out = np.zeros((actions_split.shape[0], 32), dtype=np.float32) # (H, 32)
                actions_out[:, :24] = actions_split  # fill first 24, rest = zeros

                state = actions_out[0].copy()  # first token as state
                actions = actions_out
            else:
                # sequence of states for actions
                actions = np.stack([self._state_vec(f, t0 + dt) for dt in range(self.H)], axis=0)  # (H, D)
                actions = actions[:, :32]
                state   = actions[0].copy()  # state at start

        # task = folder name (parent directory of the file)
        task = os.path.basename(os.path.dirname(h5_path))

        return {
            "image":   image,          # uint8, (H,W,3)
            "state":   state.astype(np.float32),      # (D,)
            "actions": actions.astype(np.float32),    # (H,D)
            "task":    task,
        }
############################################
############################################
# RUN BELOW TO TEST THE DATASET
############################################
############################################

# import os
# import random
# import cv2
# from pathlib import Path
# import numpy as np
# import torch

# # === configuration ===
# root_dir = "/iris/projects/humanoid/dataset/ego_dex"   # <-- change to your dataset root
# save_dir = Path("dataset_test_images")
# save_dir.mkdir(exist_ok=True)

# # === create dataset ===
# ds = EgoDexSeqDataset(
#     root_dir=root_dir,
#     action_horizon=5,        # example horizon
#     image_size=(224, 224),
#     state_format="ego",      # or "ego"
#     window_stride=1,
#     traj_per_task = 1
# )

# print(f"Dataset length: {len(ds)} samples")

# # === test retrieval ===
# num_samples_to_test = 5
# indices = random.sample(range(len(ds)), num_samples_to_test)

# for idx in indices:
#     sample = ds[idx]
#     image = sample["image"]
#     state = sample["state"]
#     actions = sample["actions"]
#     task = sample["task"]

#     print(f"Sample {idx}:")
#     print(f"  Image shape:   {image.shape}  dtype={image.dtype}")
#     print(f"  State shape:   {state.shape}")
#     print(f"  Actions shape: {actions.shape}")
#     print(f"  Task:          {task}")

#     # save image as RGB
#     img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     img_bgr_u8 = np.clip(img_bgr * 255.0, 0, 255).astype(np.uint8)
#     save_path = save_dir / f"sample_{idx}_{task}.png"
#     cv2.imwrite(str(save_path), img_bgr_u8)
#     print(f"  Saved image to {save_path}")

# print(f"\nDone! Check images in {save_dir.resolve()}")