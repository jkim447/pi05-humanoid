'''
This script is to calculate hand joint positions for the Tesollo hand model.
It takes in the egodex data and timestamp and hand type (left or right) as inputs,
and outputs the calculated hand joint positions.
'''
#!/usr/bin/env python3
####################################
# IMPORTS
####################################
import os
import csv
import numpy as np
from scipy.spatial.transform import Rotation as R


####################################
# CONSTANTS
####################################
# Camera extrinsic (camera pose in base/world) = T_base_cam0
T_base_cam0 = np.array([
    [ 0.01988061, -0.43758429,  0.89895759,  0.14056752],
    [-0.99969330,  0.00457983,  0.02433772,  0.02539622],
    [-0.01476688, -0.89916573, -0.43735903,  0.43713101],
    [ 0.0,         0.0,         0.0,         1.0       ]
], dtype=np.float32)

LEFT_HAND_25 = [
    "leftThumbKnuckle","leftThumbIntermediateBase","leftThumbIntermediateTip","leftThumbTip",
    "leftIndexFingerMetacarpal","leftIndexFingerKnuckle","leftIndexFingerIntermediateBase","leftIndexFingerIntermediateTip","leftIndexFingerTip",
    "leftMiddleFingerMetacarpal","leftMiddleFingerKnuckle","leftMiddleFingerIntermediateBase","leftMiddleFingerIntermediateTip","leftMiddleFingerTip",
    "leftRingFingerMetacarpal","leftRingFingerKnuckle","leftRingFingerIntermediateBase","leftRingFingerIntermediateTip","leftRingFingerTip",
    "leftLittleFingerMetacarpal","leftLittleFingerKnuckle","leftLittleFingerIntermediateBase","leftLittleFingerIntermediateTip","leftLittleFingerTip",
]
RIGHT_HAND_25 = [
    "rightThumbKnuckle","rightThumbIntermediateBase","rightThumbIntermediateTip","rightThumbTip",
    "rightIndexFingerMetacarpal","rightIndexFingerKnuckle","rightIndexFingerIntermediateBase","rightIndexFingerIntermediateTip","rightIndexFingerTip",
    "rightMiddleFingerMetacarpal","rightMiddleFingerKnuckle","rightMiddleFingerIntermediateBase","rightMiddleFingerIntermediateTip","rightMiddleFingerTip",
    "rightRingFingerMetacarpal","rightRingFingerKnuckle","rightRingFingerIntermediateBase","rightRingFingerIntermediateTip","rightRingFingerTip",
    "rightLittleFingerMetacarpal","rightLittleFingerKnuckle","rightLittleFingerIntermediateBase","rightLittleFingerIntermediateTip","rightLittleFingerTip",
]

EXPECTED_LEFT_JOINT_ORDER = [
    "lj_dg_1_1","lj_dg_1_2","lj_dg_1_3","lj_dg_1_4",
    "lj_dg_2_1","lj_dg_2_2","lj_dg_2_3","lj_dg_2_4",
    "lj_dg_3_1","lj_dg_3_2","lj_dg_3_3","lj_dg_3_4",
    "lj_dg_4_1","lj_dg_4_2","lj_dg_4_3","lj_dg_4_4",
    "lj_dg_5_1","lj_dg_5_2","lj_dg_5_3","lj_dg_5_4"
]
EXPECTED_RIGHT_JOINT_ORDER = [
    "rj_dg_1_1","rj_dg_1_2","rj_dg_1_3","rj_dg_1_4",
    "rj_dg_2_1","rj_dg_2_2","rj_dg_2_3","rj_dg_2_4",
    "rj_dg_3_1","rj_dg_3_2","rj_dg_3_3","rj_dg_3_4",
    "rj_dg_4_1","rj_dg_4_2","rj_dg_4_3","rj_dg_4_4",
    "rj_dg_5_1","rj_dg_5_2","rj_dg_5_3","rj_dg_5_4"
]

JOINT_LIMITS = {
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

####################################
# HELPER FUNCTIONS
####################################

def inv_SE3(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 homogeneous transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=T.dtype)
    Ti[:3, :3] = R.T
    Ti[:3, 3]  = -R.T @ t
    return Ti

def se3_from_pos_quat(pos_xyz, quat_xyzw):
    """Build SE(3) matrix from position and quaternion (xyzw)."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix().astype(np.float32)
    T[:3, 3]  = np.asarray(pos_xyz, dtype=np.float32)
    return T

def world_to_cam0(T_world_obj: np.ndarray, T_world_cam0: np.ndarray) -> np.ndarray:
    """Express obj in cam0 frame: T_cam0_obj = inv(T_world_cam0) @ T_world_obj."""
    return inv_SE3(T_world_cam0) @ T_world_obj

def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c,-s],
                     [0, s, c]], dtype=np.float32)

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0,  s],
                     [ 0, 1,  0],
                     [-s, 0,  c]], dtype=np.float32)

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c,-s, 0],
                     [ s, c, 0],
                     [ 0, 0, 1]], dtype=np.float32)



####################################
# JOINT CALCULATION FUNCTIONS
####################################

def angle_between(v1, v2):
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    v1 /= n1; v2 /= n2
    return float(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

def compute_finger_angles(finger_mat):
    p0, p1, p2, p3, p4 = finger_mat
    v0, v1, v2, v3 = p1-p0, p2-p1, p3-p2, p4-p3
    palm_normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # +X
    v0_zy = v0.copy(); v0_zy[0] = 0.0
    v1_zy = v1.copy(); v1_zy[0] = 0.0
    if np.linalg.norm(v0_zy) == 0 or np.linalg.norm(v1_zy) == 0:
        theta_mcp_abd = 0.0
    else:
        v0_zy /= np.linalg.norm(v0_zy)
        v1_zy /= np.linalg.norm(v1_zy)
        ang = np.arccos(np.clip(np.dot(v0_zy, v1_zy), -1.0, 1.0))
        cross = np.cross(v0_zy, v1_zy)
        theta_mcp_abd = float(np.sign(np.dot(cross, palm_normal)) * ang)
    return {
        'mcp_abd_rad': theta_mcp_abd,
        'mcp_flex_rad': angle_between(v0, v1),
        'pip_rad': angle_between(v1, v2),
        'dip_rad': angle_between(v2, v3),
    }

def compute_thumb_angles(finger_mat, hand):
    p0, p1, p2, p3, p4 = finger_mat
    v0, v1, v2, v3 = p1-p0, p2-p1, p3-p2, p4-p3
    v1_zy = v1.copy(); v1_zy[0] = 0.0
    if np.linalg.norm(v1_zy) == 0:
        theta_mcp_0 = 0.0
    else:
        v1_zy /= np.linalg.norm(v1_zy)
        if hand == 'left':
            theta_mcp_0 = np.radians(30.0) - angle_between(v1_zy, [0.0,-1.0,0.0])
        else:
            theta_mcp_0 = -np.radians(30.0) + angle_between(v1_zy, [0.0, 1.0,0.0])
    v0_xy = v0.copy(); v0_xy[2] = 0.0
    if np.linalg.norm(v0_xy) == 0:
        theta_mcp_1 = np.radians(90.0)
    else:
        v0_xy /= np.linalg.norm(v0_xy)
        if hand == 'left':
            theta_mcp_1 = angle_between(v0_xy, [0.0,-1.0,0.0])
        else:
            theta_mcp_1 = -angle_between(v0_xy, [0.0, 1.0,0.0])
    theta_pip = angle_between(v1, v2)
    theta_dip = angle_between(v2, v3)
    v1_zy2 = np.array([v1[1], v1[2]], dtype=np.float32)
    v2_zy2 = np.array([v2[1], v2[2]], dtype=np.float32)
    v3_zy2 = np.array([v3[1], v3[2]], dtype=np.float32)
    if np.cross(v1_zy2, v2_zy2) < 0: theta_pip *= -1.0
    if np.cross(v2_zy2, v3_zy2) < 0: theta_dip *= -1.0
    return {'mcp_0': theta_mcp_0, 'mcp_1': theta_mcp_1, 'pip_rad': theta_pip, 'dip_rad': theta_dip}

def clamp_joint_positions(qpos, joint_names):
    q = np.array(qpos, dtype=np.float32)
    q[~np.isfinite(q)] = 0.0
    margin = np.radians(1.0)
    for i, name in enumerate(joint_names):
        lo, hi = JOINT_LIMITS[name]
        q[i] = np.clip(q[i], lo + margin, hi - margin)
    return q

####################################
# MAIN FUNCTION
####################################

def hand_joint_cmd20_from_h5(f, name: str, t: int):
    """
    f: the episode file (e.g,. hdf5py file)
    name: the transformation you want (e.g., leftHand, rightHand)
    t: the time index

    return:
    np.ndarray of shape (20,) representing the 20 joint commands for left hand or the right hand (depending on what 'name' is)
    """
    # Get reference camera pose at t=0 (world frame)
    T_world_cam0 = f["transforms"]["camera"][t]

    # Determine which hand we're working with
    if name == "leftHand":
        hand = 'left'
        keypoint_names = LEFT_HAND_25  
        joint_order = EXPECTED_LEFT_JOINT_ORDER
    elif name == "rightHand":
        hand = 'right'
        keypoint_names = RIGHT_HAND_25
        joint_order = EXPECTED_RIGHT_JOINT_ORDER

    # Step 1: Get wrist pose from HDF5 (world frame) and convert to cam0 frame
    T_world_wrist = f["transforms"][name][t]
    T_cam0_wrist = world_to_cam0(T_world_wrist, T_world_cam0)

    # Step 2: Convert cam0 to base frame using T_base_cam0
    T_base_wrist = T_base_cam0 @ T_cam0_wrist

    # Step 3: Load keypoints from HDF5 (world frame) and convert to cam0 frame
    keypoints_cam0 = []
    for kp_name in keypoint_names:
        T_world_kp = f["transforms"][kp_name][t]
        T_cam0_kp = world_to_cam0(T_world_kp, T_world_cam0)
        keypoints_cam0.append(T_cam0_kp[:3, 3])  # Extract position
    keypoints_cam0 = np.array(keypoints_cam0, dtype=np.float32)  # Shape: (24, 3)

    # Step 4: Convert keypoints from cam0 to base frame
    R_base_cam = T_base_cam0[:3, :3]
    t_base_cam = T_base_cam0[:3, 3]
    keypoints_base = (R_base_cam @ keypoints_cam0.T).T + t_base_cam

    # Step 5: Transform keypoints to local wrist frame
    Rw = T_base_wrist[:3, :3]
    tw = T_base_wrist[:3, 3]
    keypoints_local = (Rw.T @ (keypoints_base - tw).T).T

    # Step 6: Apply hand-specific rotation conventions for angle computation
    if hand == 'left':
        Ry = rot_y(+np.pi/2)
        Rz_after = rot_z(+np.pi/2)
        Rc_angles = Ry @ Rz_after
    else:
        Ry = rot_y(-np.pi/2)
        Rz_after = rot_z(-np.pi/2)
        Rc_angles = Ry @ Rz_after 


    keypoints_local_conv = (Rc_angles.T @ keypoints_local.T).T

    # Step 7: Prepend wrist (origin) as first point for angle computation
    keypoints_with_wrist = np.vstack([np.zeros((1, 3), dtype=np.float32), keypoints_local_conv])

    # Step 8: Extract finger segments (indices assume 25 points with wrist at index 0)
    # Mapping: thumb at 0-4, index at 5-9, middle at 10-14, ring at 15-19, pinky at 20-24
    L_idx = {
        "thumb": (0, 1, 2, 3, 4),
        "index": (5, 6, 7, 8, 9),
        "middle": (10, 11, 12, 13, 14),
        "ring": (15, 16, 17, 18, 19),
        "pinky": (20, 21, 22, 23, 24)
    }

    thumb_pts = keypoints_with_wrist[list(L_idx["thumb"]), :]
    index_pts = keypoints_with_wrist[list(L_idx["index"]), :]
    middle_pts = keypoints_with_wrist[list(L_idx["middle"]), :]
    ring_pts = keypoints_with_wrist[list(L_idx["ring"]), :]
    pinky_pts = keypoints_with_wrist[list(L_idx["pinky"]), :]

    # Step 9: Compute angles
    thumb_angles = compute_thumb_angles(thumb_pts, hand)
    index_angles = compute_finger_angles(index_pts)
    middle_angles = compute_finger_angles(middle_pts)
    ring_angles = compute_finger_angles(ring_pts)
    pinky_angles = compute_finger_angles(pinky_pts)

    # Step 10: Build joint command array (20 joints)
    qpos = np.zeros(20, dtype=np.float32)
    idx = {n: i for i, n in enumerate(joint_order)}

    if hand == 'left':
        # Thumb
        qpos[idx["lj_dg_1_1"]] = thumb_angles['mcp_0']
        qpos[idx["lj_dg_1_2"]] = thumb_angles['mcp_1']
        qpos[idx["lj_dg_1_3"]] = thumb_angles['pip_rad']
        qpos[idx["lj_dg_1_4"]] = thumb_angles['dip_rad']

        # Index
        qpos[idx["lj_dg_2_1"]] = index_angles['mcp_abd_rad'] + np.radians(7)
        qpos[idx["lj_dg_2_2"]] = index_angles['mcp_flex_rad']
        qpos[idx["lj_dg_2_3"]] = index_angles['pip_rad']
        qpos[idx["lj_dg_2_4"]] = index_angles['dip_rad']

        # Middle
        qpos[idx["lj_dg_3_1"]] = middle_angles['mcp_abd_rad']
        qpos[idx["lj_dg_3_2"]] = middle_angles['mcp_flex_rad']
        qpos[idx["lj_dg_3_3"]] = middle_angles['pip_rad']
        qpos[idx["lj_dg_3_4"]] = middle_angles['dip_rad']

        # Ring
        qpos[idx["lj_dg_4_1"]] = ring_angles['mcp_abd_rad']
        qpos[idx["lj_dg_4_2"]] = ring_angles['mcp_flex_rad']
        qpos[idx["lj_dg_4_3"]] = ring_angles['pip_rad']
        qpos[idx["lj_dg_4_4"]] = ring_angles['dip_rad']

        # Pinky
        pinky_mcp_0 = pinky_angles['mcp_abd_rad']
        if pinky_mcp_0 > 0:
            pinky_mcp_0 = -0.05
            qpos[idx["lj_dg_5_1"]] = -pinky_angles['mcp_abd_rad']
        qpos[idx["lj_dg_5_2"]] = pinky_mcp_0
        qpos[idx["lj_dg_5_3"]] = pinky_angles['pip_rad']
        qpos[idx["lj_dg_5_4"]] = pinky_angles['dip_rad']

    else:  # right hand
        # Thumb
        qpos[idx["rj_dg_1_1"]] = thumb_angles['mcp_0']
        qpos[idx["rj_dg_1_2"]] = thumb_angles['mcp_1']
        qpos[idx["rj_dg_1_3"]] = thumb_angles['pip_rad']
        qpos[idx["rj_dg_1_4"]] = thumb_angles['dip_rad']

        # Index
        qpos[idx["rj_dg_2_1"]] = index_angles['mcp_abd_rad']
        qpos[idx["rj_dg_2_2"]] = index_angles['mcp_flex_rad']
        qpos[idx["rj_dg_2_3"]] = index_angles['pip_rad']
        qpos[idx["rj_dg_2_4"]] = index_angles['dip_rad']

        # Middle
        qpos[idx["rj_dg_3_1"]] = middle_angles['mcp_abd_rad']
        qpos[idx["rj_dg_3_2"]] = middle_angles['mcp_flex_rad']
        qpos[idx["rj_dg_3_3"]] = middle_angles['pip_rad']
        qpos[idx["rj_dg_3_4"]] = middle_angles['dip_rad']

        # Ring
        qpos[idx["rj_dg_4_1"]] = ring_angles['mcp_abd_rad']
        qpos[idx["rj_dg_4_2"]] = ring_angles['mcp_flex_rad'] * 1.1
        qpos[idx["rj_dg_4_3"]] = ring_angles['pip_rad']
        qpos[idx["rj_dg_4_4"]] = ring_angles['dip_rad']

        # Pinky
        pinky_mcp_0_r = pinky_angles['mcp_abd_rad']
        if pinky_mcp_0_r < 0:
            pinky_mcp_0_r = 0.05
            qpos[idx["rj_dg_5_1"]] = -pinky_angles['mcp_abd_rad']
        qpos[idx["rj_dg_5_2"]] = pinky_mcp_0_r
        qpos[idx["rj_dg_5_3"]] = pinky_angles['pip_rad']
        qpos[idx["rj_dg_5_4"]] = pinky_angles['dip_rad']

    # Step 11: Clamp to joint limits
    qpos = clamp_joint_positions(qpos, joint_order)

    return qpos