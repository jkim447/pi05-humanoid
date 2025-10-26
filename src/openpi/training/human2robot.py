# openpi/training/human2robot.py
import numpy as np

# ---- constants (kept internal) ----
_LEFT_HAND_24 = [
    "leftThumbKnuckle","leftThumbIntermediateBase","leftThumbIntermediateTip","leftThumbTip",
    "leftIndexFingerMetacarpal","leftIndexFingerKnuckle","leftIndexFingerIntermediateBase","leftIndexFingerIntermediateTip","leftIndexFingerTip",
    "leftMiddleFingerMetacarpal","leftMiddleFingerKnuckle","leftMiddleFingerIntermediateBase","leftMiddleFingerIntermediateTip","leftMiddleFingerTip",
    "leftRingFingerMetacarpal","leftRingFingerKnuckle","leftRingFingerIntermediateBase","leftRingFingerIntermediateTip","leftRingFingerTip",
    "leftLittleFingerMetacarpal","leftLittleFingerKnuckle","leftLittleFingerIntermediateBase","leftLittleFingerIntermediateTip","leftLittleFingerTip",
]
_RIGHT_HAND_24 = [
    "rightThumbKnuckle","rightThumbIntermediateBase","rightThumbIntermediateTip","rightThumbTip",
    "rightIndexFingerMetacarpal","rightIndexFingerKnuckle","rightIndexFingerIntermediateBase","rightIndexFingerIntermediateTip","rightIndexFingerTip",
    "rightMiddleFingerMetacarpal","rightMiddleFingerKnuckle","rightMiddleFingerIntermediateBase","rightMiddleFingerIntermediateTip","rightMiddleFingerTip",
    "rightRingFingerMetacarpal","rightRingFingerKnuckle","rightRingFingerIntermediateBase","rightRingFingerIntermediateTip","rightRingFingerTip",
    "rightLittleFingerMetacarpal","rightLittleFingerKnuckle","rightLittleFingerIntermediateBase","rightLittleFingerIntermediateTip","rightLittleFingerTip",
]

_EXPECTED_L = [
    "lj_dg_1_1","lj_dg_1_2","lj_dg_1_3","lj_dg_1_4",
    "lj_dg_2_1","lj_dg_2_2","lj_dg_2_3","lj_dg_2_4",
    "lj_dg_3_1","lj_dg_3_2","lj_dg_3_3","lj_dg_3_4",
    "lj_dg_4_1","lj_dg_4_2","lj_dg_4_3","lj_dg_4_4",
    "lj_dg_5_1","lj_dg_5_2","lj_dg_5_3","lj_dg_5_4"
]
_EXPECTED_R = [
    "rj_dg_1_1","rj_dg_1_2","rj_dg_1_3","rj_dg_1_4",
    "rj_dg_2_1","rj_dg_2_2","rj_dg_2_3","rj_dg_2_4",
    "rj_dg_3_1","rj_dg_3_2","rj_dg_3_3","rj_dg_3_4",
    "rj_dg_4_1","rj_dg_4_2","rj_dg_4_3","rj_dg_4_4",
    "rj_dg_5_1","rj_dg_5_2","rj_dg_5_3","rj_dg_5_4"
]

_JOINT_LIMITS = {
    "lj_dg_1_1": (-0.890, 0.384), "lj_dg_1_2": (0.0, 3.142), "lj_dg_1_3": (-1.571, 1.571), "lj_dg_1_4": (-1.571, 1.571),
    "lj_dg_2_1": (-0.611, 0.419), "lj_dg_2_2": (0.0, 2.007), "lj_dg_2_3": (-1.571, 1.571), "lj_dg_2_4": (-1.571, 1.571),
    "lj_dg_3_1": (-0.611, 0.611), "lj_dg_3_2": (0.0, 1.955), "lj_dg_3_3": (-1.571, 1.571), "lj_dg_3_4": (-1.571, 1.571),
    "lj_dg_4_1": (-0.419, 0.611), "lj_dg_4_2": (0.0, 1.902), "lj_dg_4_3": (-1.571, 1.571), "lj_dg_4_4": (-1.571, 1.571),
    "lj_dg_5_1": (-1.047, 0.017), "lj_dg_5_2": (-0.611, 0.419), "lj_dg_5_3": (-1.571, 1.571), "lj_dg_5_4": (-1.571, 1.571),
    "rj_dg_1_1": (-0.384, 0.890), "rj_dg_1_2": (-3.142, 0.0), "rj_dg_1_3": (-1.571, 1.571), "rj_dg_1_4": (-1.571, 1.571),
    "rj_dg_2_1": (-0.419, 0.611), "rj_dg_2_2": (0.0, 2.007), "rj_dg_2_3": (-1.571, 1.571), "rj_dg_2_4": (-1.571, 1.571),
    "rj_dg_3_1": (-0.611, 0.611), "rj_dg_3_2": (0.0, 1.955), "rj_dg_3_3": (-1.571, 1.571), "rj_dg_3_4": (-1.571, 1.571),
    "rj_dg_4_1": (-0.611, 0.419), "rj_dg_4_2": (0.0, 1.902), "rj_dg_4_3": (-1.571, 1.571), "rj_dg_4_4": (-1.571, 1.571),
    "rj_dg_5_1": (-0.017, 1.047), "rj_dg_5_2": (-0.419, 0.611), "rj_dg_5_3": (-1.571, 1.571), "rj_dg_5_4": (-1.571, 1.571),
}

# ---- tiny math helpers (internal) ----
def _rot_y(th): c,s=np.cos(th),np.sin(th); return np.array([[c,0,s],[0,1,0],[-s,0,c]],np.float32)
def _rot_z(th): c,s=np.cos(th),np.sin(th); return np.array([[c,-s,0],[s,c,0],[0,0,1]],np.float32)

def _angle_between(v1, v2):
    v1, v2 = np.asarray(v1,np.float32), np.asarray(v2,np.float32)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1==0 or n2==0: return 0.0
    v1/=n1; v2/=n2
    return float(np.arccos(np.clip(np.dot(v1,v2), -1.0, 1.0)))

def _compute_finger_angles(f):
    p0,p1,p2,p3,p4 = f
    v0,v1,v2,v3 = p1-p0, p2-p1, p3-p2, p4-p3
    palm_normal = np.array([1,0,0],np.float32)
    v0_zy, v1_zy = v0.copy(), v1.copy(); v0_zy[0]=0; v1_zy[0]=0
    if np.linalg.norm(v0_zy)==0 or np.linalg.norm(v1_zy)==0:
        theta_mcp_abd = 0.0
    else:
        v0_zy/=np.linalg.norm(v0_zy); v1_zy/=np.linalg.norm(v1_zy)
        ang = np.arccos(np.clip(np.dot(v0_zy, v1_zy), -1.0, 1.0))
        theta_mcp_abd = float(np.sign(np.dot(np.cross(v0_zy,v1_zy),palm_normal))*ang)
    return {
        "mcp_abd_rad": theta_mcp_abd,
        "mcp_flex_rad": _angle_between(v0, v1),
        "pip_rad": _angle_between(v1, v2),
        "dip_rad": _angle_between(v2, v3),
    }

def _compute_thumb_angles(f, hand):
    p0,p1,p2,p3,p4 = f
    v0,v1,v2,v3 = p1-p0, p2-p1, p3-p2, p4-p3
    v1_zy = v1.copy(); v1_zy[0]=0.0
    if np.linalg.norm(v1_zy)==0: theta_mcp_0=0.0
    else:
        v1_zy/=np.linalg.norm(v1_zy)
        theta_mcp_0 = (np.radians(30.0) - _angle_between(v1_zy,[0,-1,0])) if hand=="left" else (-np.radians(30.0) + _angle_between(v1_zy,[0,1,0]))
    v0_xy = v0.copy(); v0_xy[2]=0.0
    theta_mcp_1 = np.radians(90.0) if np.linalg.norm(v0_xy)==0 else (_angle_between(v0_xy,[0,-1,0]) if hand=="left" else -_angle_between(v0_xy,[0,1,0]))
    theta_pip = _angle_between(v1, v2)
    theta_dip = _angle_between(v2, v3)
    v1_zy2 = np.array([v1[1], v1[2]], np.float32)
    v2_zy2 = np.array([v2[1], v2[2]], np.float32)
    v3_zy2 = np.array([v3[1], v3[2]], np.float32)
    if np.cross(v1_zy2, v2_zy2) < 0: theta_pip *= -1.0
    if np.cross(v2_zy2, v3_zy2) < 0: theta_dip *= -1.0
    return {"mcp_0": theta_mcp_0, "mcp_1": theta_mcp_1, "pip_rad": theta_pip, "dip_rad": theta_dip}

def _clamp(q, names):
    q = np.array(q, np.float32); q[~np.isfinite(q)] = 0.0
    margin = np.radians(1.0)
    for i,n in enumerate(names):
        lo, hi = _JOINT_LIMITS[n]
        q[i] = np.clip(q[i], lo+margin, hi-margin)
    return q

# ---- public API ----
def hand_joint_cmd20_from_h5(f, t: int, side: str) -> np.ndarray:
    """
    Compute 20-dof robot-hand joint command from MANO-like keypoints in `f["transforms"]`.
    - Uses wrist-local geometry only (robust to camera vs world).
    - `side` in {"left","right"}.
    """
    wrist = "leftHand" if side=="left" else "rightHand"
    names24 = _LEFT_HAND_24 if side=="left" else _RIGHT_HAND_24

    # wrist pose (world)
    T_wrist = f["transforms"][wrist][t]
    Rw, tw  = T_wrist[:3,:3], T_wrist[:3,3]

    # collect 24 points (world) -> wrist-local
    pts = []
    for n in names24:
        if n not in f["transforms"]:
            pts.append([0.0,0.0,0.0])
        else:
            Tw = f["transforms"][n][t]
            p  = Tw[:3,3]
            pts.append( (Rw.T @ (p - tw)).astype(np.float32) )
    pts = np.asarray(pts, np.float32)                   # (24,3)

    # angle convention rotation (matches your script)
    Ry_left, Ry_right = _rot_y(+np.pi/2), _rot_y(-np.pi/2)
    Rz_after = _rot_z(+np.pi/2)
    R_y_final = _rot_z(np.pi)
    Rc = (Ry_left @ Rz_after) if side=="left" else (Ry_right @ Rz_after @ R_y_final)
    pts_conv = (Rc.T @ pts.T).T                         # (24,3)

    # prepend wrist (as "CMC")
    pts25 = np.vstack([np.zeros((1,3), np.float32), pts_conv])

    # per-finger slices
    idx = {"index":(5,6,7,8,9),"middle":(10,11,12,13,14),"ring":(15,16,17,18,19),
           "pinky":(20,21,22,23,24), "thumb":(0,1,2,3,4)}
    a_index  = _compute_finger_angles(pts25[list(idx["index"])])
    a_middle = _compute_finger_angles(pts25[list(idx["middle"])])
    a_ring   = _compute_finger_angles(pts25[list(idx["ring"])])
    a_pinky  = _compute_finger_angles(pts25[list(idx["pinky"])])
    a_thumb  = _compute_thumb_angles (pts25[list(idx["thumb"])], side)

    # fill in qpos
    if side=="left":
        names = _EXPECTED_L; imap = {n:i for i,n in enumerate(names)}
        q = np.zeros(20, np.float32)
        q[imap["lj_dg_2_1"]] = a_index["mcp_abd_rad"]
        q[imap["lj_dg_2_2"]] = a_index["mcp_flex_rad"]
        q[imap["lj_dg_2_3"]] = a_index["pip_rad"]
        q[imap["lj_dg_2_4"]] = a_index["dip_rad"]
        q[imap["lj_dg_3_1"]] = a_middle["mcp_abd_rad"]
        q[imap["lj_dg_3_2"]] = a_middle["mcp_flex_rad"]
        q[imap["lj_dg_3_3"]] = a_middle["pip_rad"]
        q[imap["lj_dg_3_4"]] = a_middle["dip_rad"]
        q[imap["lj_dg_4_1"]] = a_ring["mcp_abd_rad"]
        q[imap["lj_dg_4_2"]] = a_ring["mcp_flex_rad"]
        q[imap["lj_dg_4_3"]] = a_ring["pip_rad"]
        q[imap["lj_dg_4_4"]] = a_ring["dip_rad"]
        pinky0 = a_pinky["mcp_abd_rad"]; 
        if pinky0 > 0: pinky0 = -0.05
        q[imap["lj_dg_5_1"]] = a_pinky["mcp_abd_rad"]
        q[imap["lj_dg_5_2"]] = pinky0
        q[imap["lj_dg_5_3"]] = a_pinky["pip_rad"]
        q[imap["lj_dg_5_4"]] = a_pinky["dip_rad"]
        q[imap["lj_dg_1_1"]] = a_thumb["mcp_0"]
        q[imap["lj_dg_1_2"]] = a_thumb["mcp_1"]
        q[imap["lj_dg_1_3"]] = a_thumb["pip_rad"]
        q[imap["lj_dg_1_4"]] = a_thumb["dip_rad"]
        return _clamp(q, names)
    else:
        names = _EXPECTED_R; imap = {n:i for i,n in enumerate(names)}
        q = np.zeros(20, np.float32)
        q[imap["rj_dg_2_1"]] = a_index["mcp_abd_rad"]
        q[imap["rj_dg_2_2"]] = a_index["mcp_flex_rad"]
        q[imap["rj_dg_2_3"]] = a_index["pip_rad"]
        q[imap["rj_dg_2_4"]] = a_index["dip_rad"]
        q[imap["rj_dg_3_1"]] = a_middle["mcp_abd_rad"]
        q[imap["rj_dg_3_2"]] = a_middle["mcp_flex_rad"]
        q[imap["rj_dg_3_3"]] = a_middle["pip_rad"]
        q[imap["rj_dg_3_4"]] = a_middle["dip_rad"]
        q[imap["rj_dg_4_1"]] = a_ring["mcp_abd_rad"]
        q[imap["rj_dg_4_2"]] = a_ring["mcp_flex_rad"]
        q[imap["rj_dg_4_3"]] = a_ring["pip_rad"]
        q[imap["rj_dg_4_4"]] = a_ring["dip_rad"]
        pinky0 = a_pinky["mcp_abd_rad"]; 
        if pinky0 < 0: pinky0 = 0.05
        q[imap["rj_dg_5_1"]] = a_pinky["mcp_abd_rad"]
        q[imap["rj_dg_5_2"]] = pinky0
        q[imap["rj_dg_5_3"]] = a_pinky["pip_rad"]
        q[imap["rj_dg_5_4"]] = a_pinky["dip_rad"]
        q[imap["rj_dg_1_1"]] = a_thumb["mcp_0"]
        q[imap["rj_dg_1_2"]] = a_thumb["mcp_1"]
        q[imap["rj_dg_1_3"]] = a_thumb["pip_rad"]
        q[imap["rj_dg_1_4"]] = a_thumb["dip_rad"]
        return _clamp(q, names)
