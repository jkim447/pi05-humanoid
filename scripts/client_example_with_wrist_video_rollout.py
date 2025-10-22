from openpi_client import image_tools
from openpi_client import websocket_client_policy
import numpy as np
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt
if not hasattr(np, "float"):
    np.float = float
from urdfpy import URDF
from openpi.training.hand_keypoints_config import JOINT_COLOR_BGR


# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
print("im here")
client = websocket_client_policy.WebsocketClientPolicy(host="10.79.12.252", port=8000)
print("im here2")

num_steps = 1
idx = 40
# specify img path
csv_path = "/iris/projects/humanoid/openpi/scripts/demo_0_example/ee_hand.csv"
img_path = f"/iris/projects/humanoid/openpi/scripts/demo_0_example/left/{idx:06d}.jpg"
left_wrist_img_path = f"/iris/projects/humanoid/openpi/scripts/demo_0_example/left_wrist/{idx:06d}.jpg"
right_wrist_img_path = f"/iris/projects/humanoid/openpi/scripts/demo_0_example/right_wrist/{idx:06d}.jpg"
# read imgs
img = cv2.imread(img_path)
lw_img = cv2.imread(left_wrist_img_path)
rw_img = cv2.imread(right_wrist_img_path)
# convert BGR to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
lw_img = cv2.cvtColor(lw_img, cv2.COLOR_BGR2RGB)
rw_img = cv2.cvtColor(rw_img, cv2.COLOR_BGR2RGB)
# resize imgs to 224x224
img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
lw_img = cv2.resize(lw_img, (224, 224), interpolation=cv2.INTER_AREA)
rw_img = cv2.resize(rw_img, (224, 224), interpolation=cv2.INTER_AREA)
# TODO @KE make sure to flip the wrist image for the left wrist camera!
lw_img = lw_img[::-1, ::-1, :]          # numpy way
task_instruction = "vertical_pick_place"
right_hand_cols = [f"right_hand_{i}" for i in range(20)]
cv2.imwrite("left_wrist_sample.jpg", cv2.cvtColor(lw_img, cv2.COLOR_RGB2BGR))
cv2.imwrite("right_wrist_sample.jpg", cv2.cvtColor(rw_img, cv2.COLOR_RGB2BGR))
print("Saved wrist images to left_wrist_sample.jpg and right_wrist_sample.jpg")

T_BASE_TO_CAM_LEFT = np.linalg.inv(np.array([
    [0.01988061, -0.43758429,  0.89895759,  0.14056752],
    [-0.9996933,   0.00457983,  0.02433772,  0.02539622],
    [-0.01476688, -0.89916573, -0.43735903,  0.43713101],
    [0.0, 0.0, 0.0, 1.0]
]))

# --- Camera intrinsics (left), use as-is ---
K_LEFT  = np.array([[730.2571411132812, 0.0, 637.2598876953125],
                    [0.0, 730.2571411132812, 346.41082763671875],
                    [0.0, 0.0, 1.0]], dtype=np.float64)


# URDFs (load once)
_robot_l = URDF.load("/iris/projects/humanoid/act/dg_description/urdf/dg5f_left.urdf")
_robot_r = URDF.load("/iris/projects/humanoid/act/dg_description/urdf/dg5f_right.urdf")
_left_joint_names  = [j.name for j in _robot_l.joints if j.joint_type != "fixed"]
_right_joint_names = [j.name for j in _robot_r.joints if j.joint_type != "fixed"]

# Right/Left finger chains (for edges) and fingertip link names (for tips)
WRIST_L, WRIST_R = "ll_dg_base", "rl_dg_base"
FINGERS_L = {
    "Thumb": ["ll_dg_1_2","ll_dg_1_3","ll_dg_1_4","ll_dg_1_tip"],
    "Index": ["ll_dg_2_2","ll_dg_2_3","ll_dg_2_4","ll_dg_2_tip"],
    "Middle":["ll_dg_3_2","ll_dg_3_3","ll_dg_3_4","ll_dg_3_tip"],
    "Ring":  ["ll_dg_4_2","ll_dg_4_3","ll_dg_4_4","ll_dg_4_tip"],
    "Little":["ll_dg_5_2","ll_dg_5_3","ll_dg_5_4","ll_dg_5_tip"],
}
FINGERS_R = {
    "Thumb": ["rl_dg_1_2","rl_dg_1_3","rl_dg_1_4","rl_dg_1_tip"],
    "Index": ["rl_dg_2_2","rl_dg_2_3","rl_dg_2_4","rl_dg_2_tip"],
    "Middle":["rl_dg_3_2","rl_dg_3_3","rl_dg_3_4","rl_dg_3_tip"],
    "Ring":  ["rl_dg_4_2","rl_dg_4_3","rl_dg_4_4","rl_dg_4_tip"],
    "Little":["rl_dg_5_2","rl_dg_5_3","rl_dg_5_4","rl_dg_5_tip"],
}
TIP_LINKS_L = ["ll_dg_1_tip","ll_dg_2_tip","ll_dg_3_tip","ll_dg_4_tip","ll_dg_5_tip"]
TIP_LINKS_R = ["rl_dg_1_tip","rl_dg_2_tip","rl_dg_3_tip","rl_dg_4_tip","rl_dg_5_tip"]

# End-effector -> hand fixed transforms (same as your dataset code)
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
T_EE_TO_HAND_L = np.eye(4); T_EE_TO_HAND_L[:3,:3] = _R_y @ _R_z;             T_EE_TO_HAND_L[:3,3] = [ 0.00,-0.033, 0.00]
T_EE_TO_HAND_R = np.eye(4); T_EE_TO_HAND_R[:3,:3] = _R_y @ _R_z @ _R_right_z; T_EE_TO_HAND_R[:3,3] = [-0.02, 0.02, 0.025]

def _pose_to_T(pos_xyz, quat_xyzw):
    T = np.eye(4)
    T[:3,:3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = np.asarray(pos_xyz, dtype=np.float64)
    return T

def _rot_base_to_cam(R_base):
    return (T_BASE_TO_CAM_LEFT[:3,:3] @ R_base).astype(np.float32)

def _world_to_cam3(p_world3):
    ph = np.array([p_world3[0], p_world3[1], p_world3[2], 1.0], dtype=np.float64)
    return (T_BASE_TO_CAM_LEFT @ ph)[:3]

def _row_wrist_pose6d(row, side: str):
    if side == "left":
        p = [row["left_pos_x"], row["left_pos_y"], row["left_pos_z"]]
        q = [row["left_ori_x"], row["left_ori_y"], row["left_ori_z"], row["left_ori_w"]]
    else:
        p = [row["right_pos_x"], row["right_pos_y"], row["right_pos_z"]]
        q = [row["right_ori_x"], row["right_ori_y"], row["right_ori_z"], row["right_ori_w"]]
    p_cam = _world_to_cam3(np.asarray(p, dtype=np.float64))
    R_cam = _rot_base_to_cam(R.from_quat(q).as_matrix())
    ori6d = R_cam[:, :2].reshape(-1, order="F")
    return p_cam.astype(np.float32), ori6d.astype(np.float32)  # (3,), (6,)

def _read20(row, names):
    v = np.array([row[c] for c in names if c in row], dtype=np.float64)
    if v.shape[0] == 20 and np.all(np.isfinite(v)): return v
    return np.zeros(20, dtype=np.float64)

def _fk_points_world(row, side: str, kind: str = "actual"):
    # pick joint columns (fallback to cmd if actual missing)
    if side == "left":
        primary  = [f"left_{'actual_' if kind=='actual' else ''}hand_{i}"  for i in range(20)]
        fallback = [f"left_{'' if kind=='actual' else 'actual_'}hand_{i}"  for i in range(20)]
        robot, joint_names = _robot_l, _left_joint_names
        T_ee = _pose_to_T([row["left_pos_x"], row["left_pos_y"], row["left_pos_z"]],
                          [row["left_ori_x"], row["left_ori_y"], row["left_ori_z"], row["left_ori_w"]])
        T_hand = T_ee @ T_EE_TO_HAND_L
        prefix = "ll_"
    else:
        primary  = [f"right_{'actual_' if kind=='actual' else ''}hand_{i}" for i in range(20)]
        fallback = [f"right_{'' if kind=='actual' else 'actual_'}hand_{i}" for i in range(20)]
        robot, joint_names = _robot_r, _right_joint_names
        T_ee = _pose_to_T([row["right_pos_x"], row["right_pos_y"], row["right_pos_z"]],
                          [row["right_ori_x"], row["right_ori_y"], row["right_ori_z"], row["right_ori_w"]])
        T_hand = T_ee @ T_EE_TO_HAND_R
        prefix = "rl_"
    v = _read20(row, primary); 
    if not np.any(v): v = _read20(row, fallback)

    fk = robot.link_fk(cfg=dict(zip(joint_names, v.tolist())), use_names=True)
    T_fkbase_inv = np.linalg.inv(fk.get("FK_base", np.eye(4)))

    pts = {}
    for name, T_link in fk.items():
        if not name.startswith(prefix): continue
        T_world = T_hand @ (T_fkbase_inv @ T_link)
        xyz = T_world[:3, 3]
        if np.all(np.isfinite(xyz)): pts[name] = xyz
    return pts  # {link_name: (x,y,z)}

def _scale_K(K, new_W, new_H, orig_W=1280, orig_H=720):
    sx, sy = new_W / float(orig_W), new_H / float(orig_H)
    K2 = K.copy()
    K2[0,0] *= sx; K2[1,1] *= sy
    K2[0,2] *= sx; K2[1,2] *= sy
    return K2

def _link_to_mano(name: str) -> str | None:
    # Map DG names to MANO semantics used by JOINT_COLOR_BGR
    # Supports rl_/ll_ + dg numbering: *_dg_{1..5}_{2,3,4,tip}
    side = "right" if name.startswith("rl_") else "left" if name.startswith("ll_") else None
    if side is None: return None
    try:
        # e.g. rl_dg_2_3  or rl_dg_4_tip
        _, _, fid, part = name.split("_", 3)  # ["rl","dg","2","3"] or ["rl","dg","4","tip"]
    except ValueError:
        return None
    finger = {"1":"Thumb","2":"IndexFinger","3":"MiddleFinger","4":"RingFinger","5":"LittleFinger"}.get(fid)
    seg    = {"2":"Knuckle", "3":"IntermediateBase", "4":"IntermediateTip", "tip":"Tip"}.get(part)
    if finger and seg:
        prefix = "right" if side=="right" else "left"
        return f"{prefix}{finger}{seg}"
    if name.endswith("_dg_base"):     # wrist
        return "rightHand" if side=="right" else "leftHand"
    return None

def _draw_skeleton_occlusion_aware(
    img_bgr: np.ndarray,
    names: list[str],
    uv: list[tuple[int,int] | None],
    z: np.ndarray,
    edges_by_name: list[tuple[str,str]],
    color_of: dict[str, tuple[int,int,int]],
    pt_radius: int = 4,
    line_thickness: int = 2,
    edge_segments: int = 12,
) -> np.ndarray:
    """Depth-sort edges/points so nearer ones overdraw farther ones."""
    H, W = img_bgr.shape[:2]
    name_to_idx = {n:i for i,n in enumerate(names)}
    prims = []
    # edges split into small segments to approximate depth ordering
    for a,b in edges_by_name:
        if a not in name_to_idx or b not in name_to_idx: continue
        ia, ib = name_to_idx[a], name_to_idx[b]
        pa, pb = uv[ia], uv[ib]
        if (pa is None) or (pb is None): continue
        ua,va = pa; ub,vb = pb
        za,zb = float(z[ia]), float(z[ib])
        for k in range(edge_segments):
            t0 = k/edge_segments; t1=(k+1)/edge_segments
            u0 = int(round(ua*(1-t0) + ub*t0)); v0 = int(round(va*(1-t0) + vb*t0))
            u1 = int(round(ua*(1-t1) + ub*t1)); v1 = int(round(va*(1-t1) + vb*t1))
            if not (0<=u0<W and 0<=v0<H and 0<=u1<W and 0<=v1<H): 
                continue
            zmid = za*(1-(t0+t1)/2) + zb*((t0+t1)/2)
            col  = color_of.get(b, (210,210,210))
            prims.append(("edge", zmid, (u0,v0), (u1,v1), col))
    # points
    for i,p in enumerate(uv):
        if p is None: continue
        u,v = p
        if 0<=u<W and 0<=v<H:
            prims.append(("pt", float(z[i]), (u,v), color_of.get(names[i], (210,210,210))))
    # draw far→near
    prims.sort(key=lambda x: -x[1])
    out = img_bgr.copy()
    for prim in prims:
        if prim[0] == "edge":
            _,_,p0,p1,col = prim
            cv2.line(out, p0, p1, col, line_thickness, cv2.LINE_AA)
        else:
            _,_,(u,v),col = prim
            cv2.circle(out, (u,v), pt_radius, col, -1, cv2.LINE_AA)
    return out


def _overlay_hands_on_left(img_bgr, pts_L: dict|None, pts_R: dict|None):
    """
    Galaxy-style overlay: color-coded by MANO semantics + occlusion-aware,
    and draws only wrist + finger chains (no extra palm nodes).
    """
    h, w = img_bgr.shape[:2]
    Kvis = _scale_K(K_LEFT, w, h, orig_W=1280, orig_H=720)
    fx, fy, cx, cy = Kvis[0,0], Kvis[1,1], Kvis[0,2], Kvis[1,2]

    # Build names/edges limited to wrist + finger chains (same “subset” as galaxea)
    names, edges_by_name, uv, z, color_of = [], [], [], [], {}

    def _process_hand(wrist: str, fingers: dict[str, list[str]], pts_map: dict|None):
        if not pts_map: 
            return
        local_nodes = [wrist] + [n for chain in fingers.values() for n in chain]

        # edges: wrist→first link, then along each chain
        for chain in fingers.values():
            edges_by_name.append((wrist, chain[0]))
            edges_by_name.extend(zip(chain, chain[1:]))

        for name in local_nodes:
            # world->camera
            if name not in pts_map:
                uv.append(None); z.append(np.inf); names.append(name)
                continue
            Xw,Yw,Zw = pts_map[name]
            Xc,Yc,Zc = _world_to_cam3((Xw,Yw,Zw))
            if Zc <= 1e-6:
                uv.append(None); z.append(np.inf); names.append(name)
                continue
            u = int(fx*Xc/Zc + cx); v = int(fy*Yc/Zc + cy)
            if 0 <= u < w and 0 <= v < h:
                uv.append((u,v)); z.append(float(Zc))
            else:
                uv.append(None);  z.append(float(Zc))
            names.append(name)

            mano = _link_to_mano(name)
            color_of[name] = JOINT_COLOR_BGR.get(mano, (210,210,210)) if mano else (210,210,210)

    _process_hand(WRIST_L, FINGERS_L, pts_L)
    _process_hand(WRIST_R, FINGERS_R, pts_R)

    z_arr = np.asarray(z, dtype=np.float32)
    return _draw_skeleton_occlusion_aware(img_bgr, names, uv, z_arr, edges_by_name, color_of)

# --- NEW: 24D state (right hand only) ---
def _build_state24_right(row):
    # right wrist pos/orientation in LEFT camera frame
    pR, oR = _row_wrist_pose6d(row, "right")  # (3,), (6,)

    # fingertip positions from **commanded** joints (kind="cmd"), world -> left-cam
    ptsR_cmd = _fk_points_world(row, "right", kind="cmd")
    tip15 = np.concatenate([
        _world_to_cam3(ptsR_cmd[link]).astype(np.float32) if link in ptsR_cmd else np.zeros(3, np.float32)
        for link in TIP_LINKS_R  # order: thumb, index, middle, ring, pinky
    ], axis=0)  # (15,)

    return np.concatenate([pR, oR, tip15], axis=0).astype(np.float32)  # (24,)

def _build_state30(row):
    # wrist pose (left-camera frame) and 6D ori — matches galaxea
    pL, oL = _row_wrist_pose6d(row, "left")   # (3,), (6,)
    pR, oR = _row_wrist_pose6d(row, "right")  # (3,), (6,)

    # fingertips: use ACTUAL joints (then convert world -> left-cam)
    ptsL = _fk_points_world(row, "left",  kind="actual")
    ptsR = _fk_points_world(row, "right", kind="actual")

    def tip_cam_or_zero(pts_map, link):
        if link in pts_map:
            return _world_to_cam3(pts_map[link]).astype(np.float32)
        return np.zeros(3, np.float32)

    # thumbs are index 0; index fingers are index 1
    L_thumb = tip_cam_or_zero(ptsL, TIP_LINKS_L[0])
    L_index = tip_cam_or_zero(ptsL, TIP_LINKS_L[1])
    R_thumb = tip_cam_or_zero(ptsR, TIP_LINKS_R[0])
    R_index = tip_cam_or_zero(ptsR, TIP_LINKS_R[1])

    # state order (30): L pos(3)+ori6(6) + R pos(3)+ori6(6) + L tips{thumb,index}(6) + R tips{thumb,index}(6)
    return np.concatenate([pL, oL, pR, oR, L_thumb, L_index, R_thumb, R_index], axis=0).astype(np.float32)

def _project_cam_to_px(cam_xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """cam_xyz: (N,3) in camera frame -> (N,2) pixel coords (u, v)."""
    X, Y, Z = cam_xyz[:, 0], cam_xyz[:, 1], cam_xyz[:, 2]
    Z = np.clip(Z, 1e-6, None)
    u = K[0, 0] * (X / Z) + K[0, 2]
    v = K[1, 1] * (Y / Z) + K[1, 2]
    return np.stack([u, v], axis=1)

def _in_bounds(p, w, h):
    return (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)

def _draw_path(img_bgr, pts, color_bgr, thickness=2, radius=2):
    if len(pts) == 0:
        return
    for i in range(1, len(pts)):
        cv2.line(img_bgr,
                 (int(pts[i-1,0]), int(pts[i-1,1])),
                 (int(pts[i,0]),   int(pts[i,1])),
                 color_bgr, thickness, lineType=cv2.LINE_AA)
    for i in range(len(pts)):
        cv2.circle(img_bgr, (int(pts[i,0]), int(pts[i,1])), radius, color_bgr, -1, lineType=cv2.LINE_AA)

def _rot_base_to_cam(R_base):
    return (T_BASE_TO_CAM_LEFT[:3,:3] @ R_base).astype(np.float32)

def _world_to_cam3(p_world3):
    """Transform a single 3D point from robot base -> left camera frame."""
    ph = np.array([p_world3[0], p_world3[1], p_world3[2], 1.0], dtype=np.float64)
    pc = T_BASE_TO_CAM_LEFT @ ph
    return pc[:3]  # (Xc, Yc, Zc)

def _row_to_action(row):
    # --- Right wrist position in left camera frame ---
    p_world = np.array([row["right_pos_x"], row["right_pos_y"], row["right_pos_z"]], dtype=np.float64)
    p_cam = _world_to_cam3(p_world)  # (3,)

    # --- Right wrist orientation in left camera frame  ---
    rq = [row["right_ori_x"], row["right_ori_y"], row["right_ori_z"], row["right_ori_w"]]
    rR = R.from_quat(rq).as_matrix()
    R_cam = _rot_base_to_cam(rR)
    ori6d = R_cam[:, :2].reshape(-1, order="F")

    # --- Right hand command joints (20) ---
    joints20 = np.asarray([row[c] for c in right_hand_cols], dtype=np.float32)

    # --- Final action vector: [pos_cam(3), ori6d(6), joints20(20)] = 29 dims ---
    a = np.concatenate([p_cam.astype(np.float32), ori6d.astype(np.float32), joints20], axis=0)
    return a  # (29,)

# build row, overlay image, and 30D state
df = pd.read_csv(csv_path)
row = df.iloc[idx]

# 1) overlay robot hands onto the ORIGINAL image, then resize to 224
pts_L = _fk_points_world(row, "left",  kind="actual")
pts_R = _fk_points_world(row, "right", kind="actual")
img_draw_bgr = cv2.imread(img_path)                      # original BGR
img_draw_bgr = _overlay_hands_on_left(img_draw_bgr, pts_L, pts_R)
img = cv2.cvtColor(img_draw_bgr, cv2.COLOR_BGR2RGB)      # RGB
img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

# SAVE OVERLAY SAMPLES (server friendly, no visualization)
overlay_out = "hand_overlay_sample.jpg"
ok = cv2.imwrite(overlay_out, img)
print(f"Saved hand overlay to: {overlay_out} | ok={ok} | size={img_draw_bgr.shape}")


# 2) 30D state
state = _build_state30(row).astype(np.float32)
# state = _build_state24_right(row)

for step in range(num_steps):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = {
        "image": image_tools.convert_to_uint8(img),
        "wrist_image_left":  image_tools.convert_to_uint8(lw_img),
        "wrist_image_right": image_tools.convert_to_uint8(rw_img),
        "state": state,
        "prompt": task_instruction,
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]
    # state_padded = np.concatenate([state, np.zeros(2, dtype=np.float32)])  # 30 + 2 = 32
    # action_chunk = action_chunk - state_padded

    gt_idxs = list(range(idx, len(df), 3))[:25]
    gt_xyz_world = df.loc[gt_idxs, ["right_pos_x", "right_pos_y", "right_pos_z"]].to_numpy(dtype=float)
    gt_xyz = np.array([_world_to_cam3(p) for p in gt_xyz_world])  # convert to camera frame


    # 2) Policy actions: take odd timesteps only (1,3,5,...), first 3 dims are xyz
    act_xyz = action_chunk[1::2, 0:3] + state[9:12] # shape (25, 3) if action_chunk is (50, 32)
    r_thumb = action_chunk[1::2, 9:12] + state[24:27]
    r_index = action_chunk[1::2, 12:15] + state[27:30]
    # r_middle =  action_chunk[1::2, 15:18]
    # r_ring =   action_chunk[1::2, 18:21]
    # r_pinky = action_chunk[1::2, 21:24]
    

    # 3) Quick 3D plot and save
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(gt_xyz[:,0], gt_xyz[:,1], gt_xyz[:,2], marker="o", linewidth=1.5, label="GT wrist")
    ax.plot(act_xyz[:,0], act_xyz[:,1], act_xyz[:,2], marker="^", linewidth=1.5, label="Policy wrist (odd steps)")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Right Wrist Trajectory: GT vs Policy")
    ax.legend()
    plt.tight_layout()

    out_path = "wrist_traj_compare.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved 3D trajectory comparison to: {out_path}")

    img_draw = cv2.imread(img_path)  # original-resolution BGR image
    if img_draw is not None:
        h, w = img_draw.shape[:2]

        # 1) scale intrinsics to this image's size (assumes calibration at 1280x720)
        ORIG_W, ORIG_H = 1280, 720
        sx, sy = w / ORIG_W, h / ORIG_H
        K_vis = K_LEFT.copy()
        K_vis[0,0] *= sx; K_vis[1,1] *= sy
        K_vis[0,2] *= sx; K_vis[1,2] *= sy

        # 2) keep only points in front of camera (Z > 0)
        mZ_gt   = gt_xyz[:,2]  > 1e-6
        mZ_pred = act_xyz[:,2] > 1e-6
        gt_xyz_f   = gt_xyz[mZ_gt]
        act_xyz_f  = act_xyz[mZ_pred]

        # 3) project with scaled intrinsics
        px_gt   = _project_cam_to_px(gt_xyz_f,   K_vis)
        px_pred = _project_cam_to_px(act_xyz_f,  K_vis)

        # 4) in-bounds filter
        m_gt    = _in_bounds(px_gt,   w, h)
        m_pred  = _in_bounds(px_pred, w, h)
        px_gt   = px_gt[m_gt].astype(np.int32)
        px_pred = px_pred[m_pred].astype(np.int32)

        # 5) draw
        _DRAW_GREEN = (0, 255,   0)
        _DRAW_RED   = (0,   0, 255)
        _draw_path(img_draw, px_gt,   _DRAW_GREEN, thickness=2, radius=2)
        _draw_path(img_draw, px_pred, _DRAW_RED,   thickness=2, radius=2)

        # --- fingertip overlays (explicit args) ---
        def _draw_fingertip_overlay(img, cam_xyz_seq, K, color_bgr, label, offset_y):
            mZ = cam_xyz_seq[:, 2] > 1e-6
            pts3 = cam_xyz_seq[mZ]
            if len(pts3) == 0:
                return
            px = _project_cam_to_px(pts3, K)
            m_in = _in_bounds(px, img.shape[1], img.shape[0])
            px = px[m_in].astype(np.int32)
            _draw_path(img, px, color_bgr, thickness=2, radius=2)
            cv2.putText(img, label, (20, offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)

        # --- call fingertip overlays on same img_draw ---
        _DRAW_BLUE  = (255, 0, 0)
        _DRAW_YELLOW = (0, 255, 255)
        _draw_fingertip_overlay(img_draw, r_thumb, K_vis, _DRAW_BLUE, "Pred right thumb", 85)
        _draw_fingertip_overlay(img_draw, r_index, K_vis, _DRAW_YELLOW, "Pred right index", 110)


        # legend + save
        cv2.rectangle(img_draw, (10, 10), (260, 70), (255, 255, 255), -1)
        cv2.putText(img_draw, "GT right wrist",   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _DRAW_GREEN, 2, cv2.LINE_AA)
        cv2.putText(img_draw, "Pred right wrist", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _DRAW_RED,   2, cv2.LINE_AA)

        out_img = "wrist_traj_overlay.jpg"
        cv2.imwrite(out_img, img_draw)
        print(f"Saved 2D overlay to: {out_img}  |  GT pts: {len(px_gt)}  Pred pts: {len(px_pred)}")
    else:
        print(f"[warn] Could not read original image at: {img_path}")
        # Execute the actions in the environment.
    ...