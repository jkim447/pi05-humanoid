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
from openpi.training.hand_keypoints_config import LEFT_FINGERS, RIGHT_FINGERS, JOINT_COLOR_BGR
from openpi.training.human2robot import hand_joint_cmd20_from_h5

def rot_x(theta): 
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], np.float32)

def rot_y(theta): 
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], np.float32)

RC_LEFT_OFFSET  = rot_x(np.pi) @ rot_y(-np.pi/2)   # X180°, Y-90°
RC_RIGHT_OFFSET = rot_x(np.pi) @ rot_y(+np.pi/2)   # X180°, Y+90°

LEFT_MANO_21 = [
    # "leftHand",
    "leftThumbKnuckle","leftThumbIntermediateBase","leftThumbIntermediateTip","leftThumbTip",
    "leftIndexFingerKnuckle","leftIndexFingerIntermediateBase","leftIndexFingerIntermediateTip","leftIndexFingerTip",
    "leftMiddleFingerKnuckle","leftMiddleFingerIntermediateBase","leftMiddleFingerIntermediateTip","leftMiddleFingerTip",
    "leftRingFingerKnuckle","leftRingFingerIntermediateBase","leftRingFingerIntermediateTip","leftRingFingerTip",
    "leftLittleFingerKnuckle","leftLittleFingerIntermediateBase","leftLittleFingerIntermediateTip","leftLittleFingerTip",
]

RIGHT_MANO_21 = [
    # "rightHand",
    "rightThumbKnuckle","rightThumbIntermediateBase","rightThumbIntermediateTip","rightThumbTip",
    "rightIndexFingerKnuckle","rightIndexFingerIntermediateBase","rightIndexFingerIntermediateTip","rightIndexFingerTip",
    "rightMiddleFingerKnuckle","rightMiddleFingerIntermediateBase","rightMiddleFingerIntermediateTip","rightMiddleFingerTip",
    "rightRingFingerKnuckle","rightRingFingerIntermediateBase","rightRingFingerIntermediateTip","rightRingFingerTip",
    "rightLittleFingerKnuckle","rightLittleFingerIntermediateBase","rightLittleFingerIntermediateTip","rightLittleFingerTip",
]

def draw_skeleton_occlusion_aware(
    image_rgb_float: np.ndarray,
    names: list[str],
    uv: list[tuple[int,int] | None],
    z: np.ndarray,                          # (N,) camera depths, same order as names/uv
    edges_by_name: list[tuple[str,str]],    # edges as ("nameA","nameB")
    color_of: dict[str, tuple[int,int,int]],# name -> BGR color
    *,
    pt_radius: int = 6,
    line_thickness: int = 3,
    edge_segments: int = 12,
) -> np.ndarray:
    """Global depth sort of edge segments + points (far→near). Returns float RGB [0,1]."""
    H, W = image_rgb_float.shape[:2]
    name_to_idx = {n:i for i,n in enumerate(names)}
    # Build primitives
    prims = []
    # Edges → short segments with midpoint depth
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
            col  = color_of.get(b, (210,210,210))   # distal-node color
            prims.append(("edge", zmid, (u0,v0), (u1,v1), col))
    # Points
    for i, p in enumerate(uv):
        if p is None:
            continue
        u, v = p
        if 0 <= u < W and 0 <= v < H:
            prims.append(("pt", float(z[i]), (u,v), color_of.get(names[i], (210,210,210))))
    # Sort & draw (far → near)
    prims.sort(key=lambda x: -x[1])
    img_bgr = cv2.cvtColor((image_rgb_float).astype(np.uint8), cv2.COLOR_RGB2BGR)
    for prim in prims:
        if prim[0] == "edge":
            _, _, p0, p1, col = prim
            cv2.line(img_bgr, p0, p1, col, line_thickness, cv2.LINE_AA)
        else:
            _, _, (u,v), col = prim
            cv2.circle(img_bgr, (u,v), pt_radius, col, -1, lineType=cv2.LINE_AA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return img_rgb


def _draw_fingers_grouped(
    image_rgb_float: np.ndarray,
    uv: list[tuple[int,int] | None],
    names: list[str],
    name_to_idx: dict[str, int],
    *,
    line_thickness: int = 3,
    point_radius: int = 8,
) -> np.ndarray:
    """Draw per finger in order: pinky→ring→middle→index→thumb.
    For each finger: draw edges first, then points (so points stay visible)."""
    H, W, _ = image_rgb_float.shape
    img = (image_rgb_float * 255.0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    desired = ["Little", "Ring", "Middle", "Index", "Thumb"]

    def _draw_line(i_idx: int, j_idx: int, color_bgr: tuple[int,int,int]) -> None:
        pi = uv[i_idx] if 0 <= i_idx < len(uv) else None
        pj = uv[j_idx] if 0 <= j_idx < len(uv) else None
        if pi is None or pj is None:
            return
        ui, vi = pi; uj, vj = pj
        if 0 <= ui < W and 0 <= vi < H and 0 <= uj < W and 0 <= vj < H:
            cv2.line(img, (ui, vi), (uj, vj), color_bgr, line_thickness, cv2.LINE_AA)

    def _draw_point(k_idx: int, color_bgr: tuple[int,int,int]) -> None:
        p = uv[k_idx] if 0 <= k_idx < len(uv) else None
        if p is None:
            return
        u, v = p
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(img, (u, v), point_radius, color_bgr, thickness=-1, lineType=cv2.LINE_AA)

    for hand_prefix, fingers in (("left", LEFT_FINGERS), ("right", RIGHT_FINGERS)):
        wrist = f"{hand_prefix}Hand"
        for fname in desired:
            # find this finger's chain from the provided list
            chain = next((ch for ch in fingers if ch and ch[0].startswith(f"{hand_prefix}{fname}")), None)
            if not chain:
                continue

            # --- edges first: wrist→knuckle, then along the finger ---
            if wrist in name_to_idx and chain[0] in name_to_idx:
                color = JOINT_COLOR_BGR.get(chain[0], (210, 210, 210))
                _draw_line(name_to_idx[wrist], name_to_idx[chain[0]], color)
            for u_name, v_name in zip(chain, chain[1:]):
                if u_name in name_to_idx and v_name in name_to_idx:
                    color = JOINT_COLOR_BGR.get(v_name, (210, 210, 210))  # color by distal joint
                    _draw_line(name_to_idx[u_name], name_to_idx[v_name], color)

            # --- then points for this finger (keeps points on top of edges) ---
            for n in chain:
                if n in name_to_idx:
                    color = JOINT_COLOR_BGR.get(n, (0, 255, 255))
                    _draw_point(name_to_idx[n], color)

        # optional: draw wrist point last so it sits on top
        if wrist in name_to_idx:
            _draw_point(name_to_idx[wrist], JOINT_COLOR_BGR.get(wrist, (180, 180, 180)))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def _edges_by_finger_order(name_to_idx: dict[str, int], hand_prefix: str, fingers: list[list[str]]) -> list[tuple[int, int]]:
    """
    Build edges for one hand in this draw order:
    pinky → ring → middle → index → thumb
    Each item in `fingers` is a chain like:
    ["leftIndexFingerKnuckle","leftIndexFingerIntermediateBase","leftIndexFingerIntermediateTip","leftIndexFingerTip"]
    """
    desired = ["Little", "Ring", "Middle", "Index", "Thumb"]  # pinky first, then ring, middle, index, thumb
    # Map finger name -> chain
    by_name = {}
    for chain in fingers:
        if not chain:
            continue
        first = chain[0]
        # find which finger this chain is for using a simple substring check
        for fname in desired:
            if first.startswith(f"{hand_prefix}{fname}"):
                by_name[fname] = chain
                break

    edges: list[tuple[int, int]] = []
    wrist_name = f"{hand_prefix}Hand"
    if wrist_name not in name_to_idx:
        return edges

    # Append edges finger-by-finger in the requested order
    for fname in desired:
        chain = by_name.get(fname)
        if not chain:
            continue
        # wrist -> knuckle (first joint)
        if chain[0] in name_to_idx:
            edges.append((name_to_idx[wrist_name], name_to_idx[chain[0]]))
        # chain segments
        for u, v in zip(chain, chain[1:]):
            if u in name_to_idx and v in name_to_idx:
                edges.append((name_to_idx[u], name_to_idx[v]))
    return edges

def _distinct_colors(n: int) -> list[tuple[int, int, int]]:
    """Return n visually distinct BGR colors for OpenCV."""
    # evenly spaced hues in HSV, full sat/value; convert to BGR
    import colorsys
    cols = []
    for i in range(n):
        h = i / max(1, n)
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        cols.append((int(255*b), int(255*g), int(255*r)))  # BGR
    return cols

def _make_hand_edges(wrist_idx: int, first_finger_idx: int) -> list[tuple[int, int]]:
    """
    Build edges for one hand given indices in the *combined* list:
      wrist_idx: index of '...Hand'
      first_finger_idx: index where the 21 finger joints for that hand start
        order assumed: for each finger [Knuckle, IntermediateBase, IntermediateTip, Tip]
        fingers in order: Thumb, Index, Middle, Ring, Little
    """
    edges = []
    # connect wrist to each finger knuckle
    for k in [0, 4, 8, 12, 16]:
        edges.append((wrist_idx, first_finger_idx + k))
    # chains within each finger: 0-1-2-3, 4-5-6-7, ...
    for base in [0, 4, 8, 12, 16]:
        edges += [
            (first_finger_idx + base + 0, first_finger_idx + base + 1),
            (first_finger_idx + base + 1, first_finger_idx + base + 2),
            (first_finger_idx + base + 2, first_finger_idx + base + 3),
        ]
    return edges

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
    # TODO: make sure I'm activated during training!
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
        overlay: bool = False,    
    ):
        assert action_horizon >= 1
        self.root_dir = root_dir
        self.image_size = image_size
        self.state_format = state_format
        self.H = int(action_horizon)
        self.stride = max(1, int(window_stride))
        self.load_images = load_images
        self.overlay = overlay

        self.wrists = ["leftHand", "rightHand"]
        self.fingertips = [
            "leftThumbTip","leftIndexFingerTip","leftMiddleFingerTip","leftRingFingerTip","leftLittleFingerTip",
            "rightThumbTip","rightIndexFingerTip","rightMiddleFingerTip","rightRingFingerTip","rightLittleFingerTip",
        ]

        self.left_tip_order  = ["leftThumbTip","leftIndexFingerTip","leftMiddleFingerTip","leftRingFingerTip","leftLittleFingerTip"]
        self.right_tip_order = ["rightThumbTip","rightIndexFingerTip","rightMiddleFingerTip","rightRingFingerTip","rightLittleFingerTip"]


        # 1) collect (h5, mp4, N) per episode, with on-disk cache
        episodes: list[Episode] | None = None
        if not rebuild_index:
            episodes = _load_index(self.root_dir)
            print("successfully loaded cached index:", len(episodes) if episodes else 0, "episodes")

            # ---- pick a fixed number of episodes per task ----
            # TODO: when computing norm stats, might want to reduce to 25 or something 
            # TODO: note that if this line is active, we're using zero training data in most tasks!
            episodes_per_task = 0
            rng = random.Random(42)

            # tasks you want to fully include
            # TODO: include whatever task that is MOST important
            # TODO: when computing norm stats, comment out include all. When training though, activate it as needed
            # include_all = {"vertical_pick_place", "stack"}  # <== edit this list
            include_all = {"vertical_pick_place"}  # <== edit this list
            # include_all = {}

            def _task_name(ep: Episode) -> str:
                return os.path.basename(os.path.dirname(ep.h5))

            # group by task
            by_task: dict[str, list[Episode]] = {}
            for ep in episodes:
                by_task.setdefault(_task_name(ep), []).append(ep)

            picked: list[Episode] = []
            for task, eps in by_task.items():
                rng.shuffle(eps)
                if task in include_all:
                    picked.extend(eps)  # take all episodes
                else:
                    picked.extend(eps[:episodes_per_task])  # take capped subset

            episodes = picked

        if episodes is None:
            print("scanning dataset for (hdf5, mp4) pairs, this may take a while...")
            episodes = []
            # TODO: keep me active for training!
            part_dirs = sorted(
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d)) and (d.startswith("part") or d in ("test","extra"))
            )

            # TODO: keep it commented, for testing only!
            # part_dirs = ["/iris/projects/humanoid/dataset/ego_dex/part5"]
            # print("found part dirs:", part_dirs)

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
                                print(h5f, mp4f)
                                episodes.append(Episode(h5=h5f, mp4=mp4f, N=N))
                        except Exception:
                            continue

            # persist for future runs
            _save_index(self.root_dir, episodes)
            
        # keep tuple form used later
        self.episodes = [(ep.h5, ep.mp4, ep.N) for ep in episodes]

        index = []
        for ep_id, (_, _, N) in enumerate(self.episodes):
            last_start = N - 1  # allow starting at the final frame
            index.extend((ep_id, t0) for t0 in range(0, last_start + 1, self.stride))
        self.index: List[Tuple[int, int]] = index

    def __len__(self) -> int:
        return len(self.index)

    def _project_pts_onto_resized(self, pts_cam_xyz: np.ndarray, K: np.ndarray, W: int, H: int) -> np.ndarray:
        """pts_cam_xyz: (M,3) in camera frame; returns integer pixel coords on resized frame."""
        # Intrinsics are for 1920x1080 per README; scale to resized (W,H)
        W0, H0 = 1920.0, 1080.0
        sx, sy = W / W0, H / H0
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

        uv = []
        for x, y, z in pts_cam_xyz:
            if z <= 1e-6:        # behind camera or invalid
                uv.append(None)
                continue
            u = (fx * x / z) + cx
            v = (fy * y / z) + cy
            u_r = int(round(u * sx))
            v_r = int(round(v * sy))
            uv.append((u_r, v_r))
        return uv

    def _draw_keypoints(
        self,
        image_rgb_float: np.ndarray,
        uv: list[tuple[int,int] | None],
        edges: list[tuple[int,int]],
        point_colors_bgr: list[tuple[int,int,int]],
        *,
        edge_colors_bgr: Optional[list[tuple[int,int,int]]] = None,
        line_color_bgr: tuple[int,int,int] = (200, 200, 200),
        line_thickness: int = 2,
        point_radius: int = 6,
    ) -> np.ndarray:
        """Draw keypoints first (to keep them visible), then edges in the requested order."""
        H, W, _ = image_rgb_float.shape
        img = (image_rgb_float * 255.0).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 1) draw keypoints first
        for k, p in enumerate(uv):
            if p is None:
                continue
            u, v = p
            if 0 <= u < W and 0 <= v < H:
                color = point_colors_bgr[k] if k < len(point_colors_bgr) else (0, 255, 255)
                cv2.circle(img, (u, v), point_radius, color, thickness=-1, lineType=cv2.LINE_AA)

        # 2) then draw edges (pinky → ring → middle → index → thumb)
        for eidx, (i, j) in enumerate(edges):
            pi = uv[i] if 0 <= i < len(uv) else None
            pj = uv[j] if 0 <= j < len(uv) else None
            if pi is None or pj is None:
                continue
            ui, vi = pi
            uj, vj = pj
            if 0 <= ui < W and 0 <= vi < H and 0 <= uj < W and 0 <= vj < H:
                color = (
                    edge_colors_bgr[eidx]
                    if (edge_colors_bgr is not None and eidx < len(edge_colors_bgr))
                    else line_color_bgr
                )
                cv2.line(img, (ui, vi), (uj, vj), color, line_thickness, cv2.LINE_AA)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0


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

    # ---- IO helpers ----
    def _read_rgb(self, mp4_path: str, t: int) -> np.ndarray:
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            cap.release()
            return np.zeros((1080, 1920, 3), dtype=np.float32)  # fallback at typical native size

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, frame_bgr = cap.read()
        cap.release()

        if not ok or frame_bgr is None:
            return np.zeros((1080, 1920, 3), dtype=np.float32)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32)     # return native resolution, no resizing


    def _pose_cam_pos6(self, f, name: str, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (pos3, rot6) of a transform in camera frame at time t."""
        T_world_cam = f["transforms"]["camera"][t]
        T_world_obj = f["transforms"][name][t]
        T_cam = _pose_world_to_cam(T_world_obj, T_world_cam)
        pos  = T_cam[:3, 3].astype(np.float32)
        rot6 = _rotmat_to_rot6d(T_cam[:3, :3])
        return pos, rot6

    def _pose_cam_pos3(self, f, name: str, t: int) -> np.ndarray:
        """pos3 in camera frame (zeros if missing)."""
        if name not in f["transforms"]:
            return np.zeros(3, np.float32)
        T_world_cam = f["transforms"]["camera"][t]
        T_world_obj = f["transforms"][name][t]
        T_cam = _pose_world_to_cam(T_world_obj, T_world_cam)
        return T_cam[:3, 3].astype(np.float32)

    def _tips_in_wrist_frame(self, f, t: int, side: str) -> np.ndarray:
        """
        Return (15,) fingertip positions [thumb→pinky] in the wrist *local frame*.
        """
        wrist_name = "leftHand" if side == "left" else "rightHand"
        tip_names = self.left_tip_order if side == "left" else self.right_tip_order

        # wrist transform in world frame
        T_world_wrist = f["transforms"][wrist_name][t]
        R_wrist = T_world_wrist[:3, :3]
        t_wrist = T_world_wrist[:3, 3]

        tips_local = []
        for tip in tip_names:
            if tip not in f["transforms"]:
                tips_local.extend([0, 0, 0])
                continue
            T_world_tip = f["transforms"][tip][t]
            p_tip_world = T_world_tip[:3, 3]
            p_tip_local = R_wrist.T @ (p_tip_world - t_wrist)
            tips_local.extend(p_tip_local.astype(np.float32))
        return np.asarray(tips_local, dtype=np.float32)  # (15,)


    # def _hand_action24(self, f, t: int, side: str) -> np.ndarray:
    #     """(24,) = wrist pos3+rot6 + 5 fingertips (thumb→index→middle→ring→little) each xyz."""
    #     wrist = "leftHand" if side == "left" else "rightHand"
    #     pos, rot6 = self._pose_cam_pos6(f, wrist, t)
    #     tip_names = self.left_tip_order if side == "left" else self.right_tip_order
    #     tips = [self._pose_cam_pos3(f, n, t) for n in tip_names]  # 5×(3,)
    #     tips_vec = np.concatenate(tips, dtype=np.float32)         # (15,)
    #     return np.concatenate([pos, rot6, tips_vec], dtype=np.float32)  # (24,)

    def _hand_action24(self, f, t: int, side: str) -> np.ndarray:
        """
        (24,) = wrist pos3+rot6 (camera frame) + 5 fingertips (thumb→index→middle→ring→little)
        each xyz, *defined in the wrist local frame*.
        """
        wrist = "leftHand" if side == "left" else "rightHand"
        pos, rot6 = self._pose_cam_pos6(f, wrist, t)
        tip15_local = self._tips_in_wrist_frame(f, t, side)
        return np.concatenate([pos, rot6, tip15_local], dtype=np.float32)

    # def _state_both30(self, f, t: int) -> np.ndarray:
    #     """(30,) = L(pos3+rot6) + R(pos3+rot6) + L_thumb + L_index + R_thumb + R_index (each xyz)."""
    #     Lp, Lr = self._pose_cam_pos6(f, "leftHand",  t)
    #     Rp, Rr = self._pose_cam_pos6(f, "rightHand", t)
    #     L_thumb = self._pose_cam_pos3(f, "leftThumbTip",        t)
    #     L_index = self._pose_cam_pos3(f, "leftIndexFingerTip",  t)
    #     R_thumb = self._pose_cam_pos3(f, "rightThumbTip",       t)
    #     R_index = self._pose_cam_pos3(f, "rightIndexFingerTip", t)
    #     return np.concatenate([Lp, Lr, Rp, Rr, L_thumb, L_index, R_thumb, R_index], dtype=np.float32)

    def _state_both30(self, f, t: int) -> np.ndarray:
        """
        (30,) = L(pos3+rot6) + R(pos3+rot6)
                + L_thumb(3) + L_index(3) + R_thumb(3) + R_index(3)
        with thumb/index in *wrist local frames*.
        """
        Lp, Lr = self._pose_cam_pos6(f, "leftHand",  t)
        Rp, Rr = self._pose_cam_pos6(f, "rightHand", t)

        tipsL = self._tips_in_wrist_frame(f, t, "left")
        tipsR = self._tips_in_wrist_frame(f, t, "right")

        L_thumb = tipsL[0:3]
        L_index = tipsL[3:6]
        R_thumb = tipsR[0:3]
        R_index = tipsR[3:6]

        return np.concatenate([Lp, Lr, Rp, Rr, L_thumb, L_index, R_thumb, R_index], dtype=np.float32)

    # ---------- define offset helpers once near top ----------
    # replace the old signature/body
    def _pose_cam_pos6_offset(self, f, name: str, t: int, cam_t: Optional[int] = None):
        """
        Same as _pose_cam_pos6 but applies wrist orientation offsets.
        The camera frame is taken at cam_t (defaults to t if None).
        """
        cam_idx = t if cam_t is None else cam_t
        T_world_cam = f["transforms"]["camera"][cam_idx]
        T_world_obj = f["transforms"][name][t]
        T_cam = _pose_world_to_cam(T_world_obj, T_world_cam)

        pos = T_cam[:3, 3].astype(np.float32)
        Rm  = T_cam[:3, :3]

        if name == "leftHand":
            Rm = Rm @ RC_LEFT_OFFSET
        elif name == "rightHand":
            Rm = Rm @ RC_RIGHT_OFFSET

        rot6 = np.array([Rm[0,0], Rm[1,0], Rm[2,0], Rm[0,1], Rm[1,1], Rm[2,1]], np.float32)
        return pos, rot6

    def center_crop(self, img: np.ndarray, s: float = 0.68, dx: int = 0, dy: int = 0) -> np.ndarray:
        """Keep s fraction of width/height. Optional (dx,dy) shifts the crop window."""
        H, W = img.shape[:2]
        new_W, new_H = int(W * s), int(H * s)
        x1 = (W - new_W) // 2 + dx
        y1 = (H - new_H) // 2 + dy
        x1 = max(0, min(x1, W - new_W))
        y1 = max(0, min(y1, H - new_H))
        return img[y1:y1+new_H, x1:x1+new_W]

    def bottom_crop(self, img: np.ndarray, s: float = 0.80) -> np.ndarray:
        """Keep s fraction of width/height, centered horizontally, anchored to bottom."""
        H, W = img.shape[:2]
        new_W, new_H = int(W * s), int(H * s)
        x1 = (W - new_W) // 2
        y1 = max(0, H - new_H)   # start so we keep the bottom part
        return img[y1:y1+new_H, x1:x1+new_W]


    # ---- main fetch ----
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_id, t0 = self.index[idx]
        h5_path, mp4_path, N = self.episodes[ep_id]
        t0 = min(t0, N - 1)  # clamp

        if self.load_images:
            image = self._read_rgb(mp4_path, t0)
        else:
            H, W = self.image_size
            image = np.zeros((H, W, 3), dtype=np.float32)  # <— NEW: no image I/O for stats

        with h5py.File(h5_path, "r") as f:

            # ----- optional keypoint overlay (simple & clear) -----
            if self.load_images and self.overlay:
                K = np.array([[736.6339, 0., 960.],
                    [0., 736.6339, 540.],
                    [0., 0., 1.]], dtype=np.float64)

                T_world_cam = f["transforms"]["camera"][t0]

                # Include wrists + the 21 finger joints for both hands.
                # Order matters because edges are built by index.
                names = (
                    ["leftHand"] + [n for n in LEFT_MANO_21 if n in f["transforms"]] +
                    ["rightHand"] + [n for n in RIGHT_MANO_21 if n in f["transforms"]]
                )

                # Ensure both wrists exist; if not, skip overlay safely
                names = [n for n in names if n in f["transforms"]]

                # 3D -> camera frame -> pixel coords on *native* image size
                pts_cam = []
                for n in names:
                    T_w = f["transforms"][n][t0]
                    T_c = _pose_world_to_cam(T_w, T_world_cam)
                    pts_cam.append(T_c[:3, 3])
                pts_cam = np.asarray(pts_cam, dtype=np.float64)

                H0, W0 = image.shape[:2]
                uv = self._project_pts_onto_resized(pts_cam, K, W0, H0)

                # ---- occlusion-aware overlay (drop-in) ----
                # assumes you already have: names (list[str]), pts_cam (Nx3), uv (list[tuple|None]), image (RGB float [0,1])
                # IMPORTANT: do not reorder names/uv before this block.
                z = pts_cam[:, 2].astype(np.float32)   # (N,)

                # 1) Build edges by *name* (wrist→knuckle + finger chains), robust to missing joints.
                edges_by_name: list[tuple[str,str]] = []

                def _add_chain(chain: list[str]):
                    # chain like ["leftIndexFingerKnuckle","leftIndexFingerIntermediateBase",...]
                    for u, v in zip(chain, chain[1:]):
                        if (u in names) and (v in names):
                            edges_by_name.append((u, v))

                # Left hand
                if "leftHand" in names:
                    for chain in LEFT_FINGERS:
                        if chain and (chain[0] in names):
                            edges_by_name.append(("leftHand", chain[0]))  # wrist -> knuckle
                        _add_chain(chain)

                # Right hand
                if "rightHand" in names:
                    for chain in RIGHT_FINGERS:
                        if chain and (chain[0] in names):
                            edges_by_name.append(("rightHand", chain[0]))
                        _add_chain(chain)

                # 2) Per-joint colors from shared config (fallback if a name isn't in the palette)
                fallback = (210, 210, 210)
                color_of = {n: JOINT_COLOR_BGR.get(n, fallback) for n in names}

                # 3) One global depth-sorted draw for BOTH hands (edges inherit distal joint color)
                image = draw_skeleton_occlusion_aware(
                    image_rgb_float=image,
                    names=names,
                    uv=uv,
                    z=z,
                    edges_by_name=edges_by_name,
                    color_of=color_of,
                    pt_radius=12,
                    line_thickness=25,
                    edge_segments=12,
                )

                # center crop to match zed mini image view
                CROP_S = 0.80  # ≈ 82°×52° target -- we need to crop by 68 percent to match zed mini, but misses many objects
                image = self.bottom_crop(image, s=CROP_S)  # center crop (use dx,dy if you want off-center)
                                                
            # --- Actions & State (robot-style) ---
            Lp0, Lr0 = self._pose_cam_pos6_offset(f, "leftHand",  t0, cam_t=t0)
            Rp0, Rr0 = self._pose_cam_pos6_offset(f, "rightHand", t0, cam_t=t0)

            qL0 = hand_joint_cmd20_from_h5(f, t0, "left")
            qR0 = hand_joint_cmd20_from_h5(f, t0, "right")

            # actions relative to the same reference camera frame
            actions_lr = []
            for dt in range(self.H):
                t = t0 + dt
                if t < N:
                    Lp_t, Lr_t = self._pose_cam_pos6_offset(f, "leftHand",  t, cam_t=t0)
                    Rp_t, Rr_t = self._pose_cam_pos6_offset(f, "rightHand", t, cam_t=t0)

                    dLp = (Lp_t - Lp0).astype(np.float32)
                    dLr = (Lr_t - Lr0).astype(np.float32)
                    dRp = (Rp_t - Rp0).astype(np.float32)
                    dRr = (Rr_t - Rr0).astype(np.float32)

                    qLt = hand_joint_cmd20_from_h5(f, t, "left")
                    qRt = hand_joint_cmd20_from_h5(f, t, "right")

                    # djL = (qLt - qL0).astype(np.float32)
                    # djR = (qRt - qR0).astype(np.float32)

                    qLt = (qLt).astype(np.float32)
                    qRt = (qRt).astype(np.float32)

                    actions_lr.extend([
                        np.concatenate([dLp, dLr, qLt], axis=0),
                        np.concatenate([dRp, dRr, qRt], axis=0),
                    ])
                else:
                    actions_lr.extend([np.zeros(29, np.float32), np.zeros(29, np.float32)])

            actions = np.stack(actions_lr, axis=0).astype(np.float32)

            # --- State (32): L pos3+rot6, R pos3+rot6, L joints[1:8], R joints[1:8] ---
            state = np.concatenate([Lp0, Lr0, Rp0, Rr0, qL0[1:8], qR0[1:8]], axis=0).astype(np.float32)

        # task = folder name (parent directory of the file)
        task = os.path.basename(os.path.dirname(h5_path))

        # final resize to self.image_size
        out_h, out_w = self.image_size
        # TODO: uncomment me!
        image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(np.float32)

        return {
            "image":   image.astype(np.uint8),          # uint8, (H,W,3)
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
# save_dir = Path("/iris/projects/humanoid/openpi/dataset_test_images")
# save_dir.mkdir(exist_ok=True)

# # === create dataset ===
# ds = EgoDexSeqDataset(
#     root_dir=root_dir,
#     action_horizon=25,        # example horizon
#     image_size=(224, 224),
#     state_format="ego_split",      # or "ego"
#     window_stride=1,
#     traj_per_task = None,
#     overlay=True,
#     rebuild_index=False,    # set True to rescan dataset
# )

# print(f"Dataset length: {len(ds)} samples")

# # === test retrieval ===
# num_samples_to_test = 40
# indices = random.sample(range(len(ds)), num_samples_to_test)

# for idx in indices:
#     sample = ds[idx]
#     image = sample["image"]        # already uint8 RGB
#     state = sample["state"]
#     actions = sample["actions"]
#     task = sample["task"]

#     print(f"Sample {idx}:")
#     print(f"  Image shape:   {image.shape}  dtype={image.dtype}")
#     print(f"  State shape:   {state.shape}")
#     print(f"  Actions shape: {actions.shape}")
#     print(f"  Task:          {task}")

#     # Convert RGB → BGR for OpenCV saving
#     img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     save_path = save_dir / f"sample_{idx}_{task}.png"
#     cv2.imwrite(str(save_path), img_bgr)
#     print(f"  Saved image to {save_path}")