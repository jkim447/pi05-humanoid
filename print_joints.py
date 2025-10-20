import numpy as np
if not hasattr(np, "float"):
    np.float = float
from urdfpy import URDF

urdf_left_path = "/iris/projects/humanoid/act/dg_description/urdf/dg5f_left.urdf"
urdf_right_path = "/iris/projects/humanoid/act/dg_description/urdf/dg5f_right.urdf"

robot_l = URDF.load(urdf_left_path)
robot_r = URDF.load(urdf_right_path)

print("=== LEFT JOINTS ===")
for j in robot_l.joints:
    print(f"{j.name:25s}  type={j.joint_type}  parent={j.parent}  ->  child={j.child}")

print("\n=== RIGHT JOINTS ===")
for j in robot_r.joints:
    print(f"{j.name:25s}  type={j.joint_type}  parent={j.parent}  ->  child={j.child}")
