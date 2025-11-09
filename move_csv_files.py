import os
import shutil

def copy_robot_commands(src_root, dst_root):
    for demo_name in os.listdir(src_root):
        src_demo = os.path.join(src_root, demo_name)
        dst_demo = os.path.join(dst_root, demo_name)
        src_file = os.path.join(src_demo, "robot_commands.csv")
        dst_file = os.path.join(dst_demo, "robot_commands.csv")

        if os.path.isfile(src_file) and os.path.isdir(dst_demo):
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {src_file} → {dst_file}")
        else:
            print(f"Skipped: {demo_name} (missing file or folder)")

# Example usage:
src = "/iris/projects/humanoid/hamer/FINAL_OUT_HUMAN/HUMAN_BOX_PLACE_COMBO_1105"
dst = "/iris/projects/humanoid/dataset/HUMAN_BOX_PLACE_COMBO_1105"
copy_robot_commands(src, dst)
