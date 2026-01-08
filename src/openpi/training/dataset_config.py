# openpi/training/dataset_config.py
from typing import NamedTuple, Literal
import os

class DatasetEntry(NamedTuple):
    kind: Literal["egodex", "robot", "human"]
    path: str
    weight: float              # target sampling probability
    apply_custom_norm: bool
    norm_stats_path: str  # Path to normalization statistics file
    stride: int = 2
    task: str = "vertical_pick_place"
    overlay: bool = True
    mask_wrist: bool = True
    overlay_both: bool = False # option for our_human_dataset, to overlay keypoints on both hands. Exists since most data is right hand only
    both_actions: bool = False # option for our_human_dataset to return valid actions for both hands. Also exists since most data is right hand only
    ee_to_hand_left_xyz: tuple[float, float, float] | None = None
    ee_to_hand_right_xyz: tuple[float, float, float] | None = None


base_pth = "/iris/projects/humanoid/dataset/"

# TODO: make sure ONLY your desired datasets are uncommented!
# TODO: ensure the ratio of each data!
# TODO: scroll to the bottom to check, since it's a long file!
# TODO: NOTE: sometimes human has both hands in the view, so for that
# appropriately set overlay_both and both_actions to True for those
# human dataset! SUPER IMPORTANT


###############################################################################
###############################################################################
###############################################################################
# PICK PLACE REDO USE ME! 11132025
###############################################################################
###############################################################################
###############################################################################

########################################
# Non-baseline (robot + human data, hard)
########################################

# TODO: set accordingly, set False for ablation study
# overlay = True
# DATASETS = [
#     DatasetEntry(
#         kind="robot",
#         path= os.path.join(base_pth, "ROBOT_PICK_REDCUBE_1107"),
#         weight=0.5,
#         task ="vertical_pick_place",
#         stride=2,
#         overlay=overlay,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None
#     ),

#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_PICK_REDCUBE_1107",
#         weight=0.125,
#         overlay=overlay,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None,
#         # overlay_both=True, # 
#         # both_actions=True,# 
#     ),

#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_PICK_CHICKEN_1107",
#         weight=0.125,
#         overlay=overlay,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None,
#         # overlay_both=True, # 
#         # both_actions=True,# 
#     ),

#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_PICK_CORN_1107",
#         weight=0.125,
#         overlay=overlay,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None,
#         # overlay_both=True, # 
#         # both_actions=True,# 
#     ),

#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_PICK_PEPPER_1107",
#         weight=0.125,
#         overlay=overlay,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None,
#         # overlay_both=True, # 
#         # both_actions=True,# 
#     ),
# ]

# ########################################
# # Baseline (robot data only, hard)
# ########################################
# DATASETS = [
#     DatasetEntry(
#         kind="robot",
#         path= os.path.join(base_pth, "ROBOT_PICK_REDCUBE_1107"),
#         weight=1.0,
#         task ="vertical_pick_place",
#         stride=2,
#         overlay=True,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None
#     ),
# ]

###############################################################################
###############################################################################
###############################################################################
# COMPOSITION -- Box pull + pick place
###############################################################################
###############################################################################
###############################################################################

#####################################################
# YET ANOTHER NEW COMPOSITION BASELINE 11112025 USE ME
#####################################################

# DATASETS = [

#     DatasetEntry(
#         kind="robot",
#         path="/iris/projects/humanoid/dataset/ROBOT_OPEN_BOX_1111",
#         weight=0.50,
#         task ="vertical_pick_place",
#         stride=3,
#         overlay=True,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None,
#         ee_to_hand_left_xyz = [0.01, 0.02, -0.035],
#         ee_to_hand_right_xyz = [-0.01, -0.07, 0.00]

#         # TODO: add custom offsets here for good measure
#     ),

#     # below is sort cotrain data
#     DatasetEntry(
#         kind="robot",
#         path="/iris/projects/humanoid/dataset/ROBOT_SORT_RED_LEFT_1110",
#         weight=0.25,
#         task ="vertical_pick_place",
#         stride=3,
#         overlay=True,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None,
#         ee_to_hand_left_xyz = [0.01, 0.015, -0.05], # TODO: this is added for the new composition data collected 1110... instead of collect human data we collected robot data
#         ee_to_hand_right_xyz = [-0.01, -0.06, 0.015]
#     ),

#     DatasetEntry(
#         kind="robot",
#         path="/iris/projects/humanoid/dataset/ROBOT_SORT_TL_1104",
#         weight=0.25,
#         task ="vertical_pick_place",
#         stride=3,
#         overlay=True,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None,
#         ee_to_hand_left_xyz = [0.01, -0.019, -0.04], # TODO: this is added for the new composition data collected 1110... instead of collect human data we collected robot data
#         ee_to_hand_right_xyz = [-0.01, -0.005, 0.015]
#         # TODO: add custom offsets here for good measure
#     ),
# ]

#####################################################
# YET ANOTHER NEW COMPOSITION! 11112025 USE ME
#####################################################
# TODO: set false for no keypoint baseline
# overlay = True # TODO: set this option accordingly!
# DATASETS = [

# #     # below is composition data
#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_OPEN_BOX_COMBO_1111",
#         weight=0.34,
#         overlay=overlay,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None,
#         overlay_both=True, # TODO: the option to overlay both hands is set!
#         both_actions=True,# TODO: this is bimanual action data so return both valid left / right actions!
#     ),

#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_OPEN_BOX_1111",
#         weight=0.03,
#         overlay=overlay,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None,
#         overlay_both=True, # TODO: the option to overlay both hands is set!
#         both_actions=True,# TODO: this is bimanual action data so return both valid left / right actions!
#     ),

#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_SORT_TL_1104_COPY",
#         weight=0.03,
#         overlay=overlay,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None
#     ),    

#     DatasetEntry(
#         kind="robot",
#         path="/iris/projects/humanoid/dataset/ROBOT_OPEN_BOX_1111",
#         weight=0.3,
#         task ="vertical_pick_place",
#         stride=3,
#         overlay=overlay,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None,
#         ee_to_hand_left_xyz = [0.01, 0.02, -0.035],
#         ee_to_hand_right_xyz = [-0.01, -0.07, 0.00]

#         # TODO: add custom offsets here for good measure
#     ),

#     # below is sort cotrain data
#     DatasetEntry(
#         kind="robot",
#         path="/iris/projects/humanoid/dataset/ROBOT_SORT_RED_LEFT_1110",
#         weight=0.15,
#         task ="vertical_pick_place",
#         stride=3,
#         overlay=overlay,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None,
#         ee_to_hand_left_xyz = [0.01, 0.015, -0.05], # TODO: this is added for the new composition data collected 1110... instead of collect human data we collected robot data
#         ee_to_hand_right_xyz = [-0.01, -0.06, 0.015]
#     ),

#     DatasetEntry(
#         kind="robot",
#         path="/iris/projects/humanoid/dataset/ROBOT_SORT_TL_1104",
#         weight=0.15,
#         task ="vertical_pick_place",
#         stride=3,
#         overlay=overlay,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None,
#         ee_to_hand_left_xyz = [0.01, -0.019, -0.04], # TODO: this is added for the new composition data collected 1110... instead of collect human data we collected robot data
#         ee_to_hand_right_xyz = [-0.01, -0.005, 0.015]
#         # TODO: add custom offsets here for good measure
#     ),
# ]

#####################################################
# Composition fade away overlay! 01052026 USE ME
#####################################################
# TODO: set false for no keypoint baseline
overlay = True # TODO: set this option accordingly!
DATASETS = [
    DatasetEntry(
        kind="human",
        path="/iris/projects/humanoid/dataset/HUMAN_OPEN_BOX_COMBO_1111",
        weight=0.34,
        overlay=overlay,
        task = "vertical_pick_place",
        apply_custom_norm=False,
        norm_stats_path=None,
        overlay_both=True, # TODO: the option to overlay both hands is set!
        both_actions=True,# TODO: this is bimanual action data so return both valid left / right actions!
    ),

    DatasetEntry(
        kind="human",
        path="/iris/projects/humanoid/dataset/HUMAN_OPEN_BOX_1111",
        weight=0.03,
        overlay=overlay,
        task = "vertical_pick_place",
        apply_custom_norm=False,
        norm_stats_path=None,
        overlay_both=True, # TODO: the option to overlay both hands is set!
        both_actions=True,# TODO: this is bimanual action data so return both valid left / right actions!
    ),

    DatasetEntry(
        kind="human",
        path="/iris/projects/humanoid/dataset/HUMAN_SORT_TL_1104_COPY",
        weight=0.03,
        overlay=overlay,
        task = "vertical_pick_place",
        apply_custom_norm=False,
        norm_stats_path=None
    ),    

    DatasetEntry(
        kind="robot",
        path="/iris/projects/humanoid/dataset/ROBOT_OPEN_BOX_1111",
        weight=0.3,
        task ="vertical_pick_place",
        stride=3,
        overlay=overlay,
        mask_wrist=True,
        apply_custom_norm=False,
        norm_stats_path = None,
        ee_to_hand_left_xyz = [0.01, 0.02, -0.035],
        ee_to_hand_right_xyz = [-0.01, -0.07, 0.00]
    ),

    # below is sort cotrain data
    DatasetEntry(
        kind="robot",
        path="/iris/projects/humanoid/dataset/ROBOT_SORT_RED_LEFT_1110",
        weight=0.15,
        task ="vertical_pick_place",
        stride=3,
        overlay=overlay,
        mask_wrist=True,
        apply_custom_norm=False,
        norm_stats_path = None,
        ee_to_hand_left_xyz = [0.01, 0.015, -0.05], # TODO: this is added for the new composition data collected 1110... instead of collect human data we collected robot data
        ee_to_hand_right_xyz = [-0.01, -0.06, 0.015]
    ),

    DatasetEntry(
        kind="robot",
        path="/iris/projects/humanoid/dataset/ROBOT_SORT_TL_1104",
        weight=0.15,
        task ="vertical_pick_place",
        stride=3,
        overlay=overlay,
        mask_wrist=True,
        apply_custom_norm=False,
        norm_stats_path = None,
        ee_to_hand_left_xyz = [0.01, -0.019, -0.04], # TODO: this is added for the new composition data collected 1110... instead of collect human data we collected robot data
        ee_to_hand_right_xyz = [-0.01, -0.005, 0.015]
        # TODO: add custom offsets here for good measure
    ),
]


#####################################################
# NEW COMPOSITION! 10102025
#####################################################

# overlay = True
# # # TODO: NOTE: I'm using the same langauge instruction for everything for this task!
# DATASETS = [


# #     # below is composition data
#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_BOX_PLACE_COMBO_1105",
#         weight=0.30,
#         overlay=True,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None,
#         overlay_both=True, # TODO: the option to overlay both hands is set!
#         both_actions=True,# TODO: this is bimanual action data so return both valid left / right actions!
#     ),

#     # below is box pull data
#     DatasetEntry(
#         kind="robot",
#         path="/iris/projects/humanoid/dataset/ROBOT_PULL_BOX_1105",
#         weight=0.27,
#         task ="vertical_pick_place", # TODO: set correct language instructions across
#         stride=4, # TODO: NOTE: stride 4 is used for box pull!
#         overlay=True,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None,
#         ee_to_hand_left_xyz = [0.01, -0.019, -0.04], # TODO: this is added for the new composition data collected 1110... instead of collect human data we collected robot data
#         ee_to_hand_right_xyz = [-0.01, -0.005, 0.015]
#     ),

#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_PULL_BOX_1105",
#         weight=0.03,
#         overlay=True,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None,
#         overlay_both=True, # TODO: the option to overlay both hands is set!
#         both_actions=True, # TODO: this is bimanual action data so return both valid left / right actions!
#     ),

#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_PICK_REDCUBE_1107",
#         weight=0.1,
#         overlay=overlay,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None,
#         # overlay_both=True, # 
#         # both_actions=True,# 
#     ),

#     DatasetEntry(
#         kind="robot",
#         path= os.path.join(base_pth, "ROBOT_PICK_REDCUBE_1107"),
#         weight=0.30,
#         task ="vertical_pick_place",
#         stride=3,
#         overlay=overlay,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None,
#         ee_to_hand_left_xyz = [0.0, 0.03, -0.04], # TODO: this is added for the new composition data collected 1110... instead of collect human data we collected robot data
#         ee_to_hand_right_xyz = [-0.01, -0.06, 0.01]
#     ),

# ]

#####################################################
# COMPOSITION BASELINE
#####################################################

# DATASETS = [

#     # below is box pull data
#     DatasetEntry(
#         kind="robot",
#         path="/iris/projects/humanoid/dataset/ROBOT_PULL_BOX_1105",
#         weight=0.34,
#         task ="vertical_pick_place", # TODO: set correct language instructions across
#         stride=4, # TODO: NOTE: stride 4 is used for box pull!
#         overlay=True,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None
#     ),

#     # below is sort cotrain data
#     DatasetEntry(
#         kind="robot",
#         path="/iris/projects/humanoid/dataset/ROBOT_SORT_TL_1104",
#         weight=0.33,
#         task ="vertical_pick_place",
#         stride=2,
#         overlay=True,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None
#     ),

#     DatasetEntry(
#         kind="robot",
#         path="/iris/projects/humanoid/dataset/ROBOT_SORT_TR_1104",
#         weight=0.33,
#         task ="vertical_pick_place",
#         stride=2,
#         overlay=True,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None
#     ),
# ]

#####################################################
# COMPOSITION NON-BASELINE
#####################################################
# # TODO: NOTE: I'm using the same langauge instruction for everything for this task!
# DATASETS = [

# #     # below is composition data
#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_BOX_PLACE_COMBO_1105",
#         weight=0.28,
#         overlay=True,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None,
#         overlay_both=True, # TODO: the option to overlay both hands is set!
#         both_actions=True,# TODO: this is bimanual action data so return both valid left / right actions!
#     ),

#     # below is box pull data
#     DatasetEntry(
#         kind="robot",
#         path="/iris/projects/humanoid/dataset/ROBOT_PULL_BOX_1105",
#         weight=0.19,
#         task ="vertical_pick_place", # TODO: set correct language instructions across
#         stride=4, # TODO: NOTE: stride 4 is used for box pull!
#         overlay=True,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None
#     ),

#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_PULL_BOX_1105",
#         weight=0.05,
#         overlay=True,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None,
#         overlay_both=True, # TODO: the option to overlay both hands is set!
#         both_actions=True, # TODO: this is bimanual action data so return both valid left / right actions!
#     ),

#     # below is sort cotrain data
#     DatasetEntry(
#         kind="robot",
#         path="/iris/projects/humanoid/dataset/ROBOT_SORT_RED_LEFT_1110",
#         weight=0.19,
#         task ="vertical_pick_place",
#         stride=3,
#         overlay=True,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None,
#         ee_to_hand_left_xyz = [0.01, 0.015, -0.05], # TODO: this is added for the new composition data collected 1110... instead of collect human data we collected robot data
#         ee_to_hand_right_xyz = [-0.01, -0.06, 0.015]

#     ),
#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_SORT_TL_1104_COPY",
#         weight=0.05,
#         overlay=True,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None
#     ),    
#     DatasetEntry(
#         kind="robot",
#         path="/iris/projects/humanoid/dataset/ROBOT_SORT_BLUE_RIGHT_1110",
#         weight=0.19,
#         task ="vertical_pick_place",
#         stride=3,
#         overlay=True,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None,
#         ee_to_hand_left_xyz = [0.01, 0.015, -0.05], # TODO: this is added for the new composition data collected 1110... instead of collect human data we collected robot data
#         ee_to_hand_right_xyz = [-0.01, -0.06, 0.015]

#     ),
#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_SORT_TR_1104_COPY",
#         weight=0.05,
#         overlay=True,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None
#     ),
# ]

###############################################################################
###############################################################################
###############################################################################
# PICK PLACE
###############################################################################
###############################################################################
###############################################################################

#####################################################
# all pick place robot data + egodex
# TODO: NOTE: overlay is set to True!
######################################################

# DATASETS = [

#     DatasetEntry(
#         kind="egodex",
#         path="/iris/projects/humanoid/dataset/ego_dex",
#         overlay = True,
#         weight=0.6,  # no training yet
#         apply_custom_norm=False,
#         norm_stats_path=None
#     ),

#     DatasetEntry(
#         kind="robot",
#         path= os.path.join(base_pth, "ROBOT_PICK_PLACE_LEFT"),
#         weight=0.1,
#         task ="vertical_pick_place",
#         stride=2,
#         overlay=True,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None
#     ),

#     DatasetEntry(
#         kind="robot",
#         path=os.path.join(base_pth, "ROBOT_PICK_PLACE_RIGHT"),
#         weight=0.1,
#         task ="vertical_pick_place",
#         stride=2,
#         overlay=True,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None
#     ),


#     DatasetEntry(
#         kind="robot",
#         path=os.path.join(base_pth, "ROBOT_PICK_EXTRA_RIGHT_1031"),
#         weight=0.1,
#         task ="vertical_pick_place",
#         stride=2,
#         overlay=True,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None
#     ),


#     DatasetEntry(
#         kind="robot",
#         path=os.path.join(base_pth, "ROBOT_PICK_PLACE_INTER_1030"),
#         weight=0.1,
#         task ="vertical_pick_place",
#         stride=2,
#         overlay=True,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None
#     ),
# ]

#####################################################
# baseline, robot data only at all regions 11052025
# TODO: NOTE: overlay is set to False!
######################################################


# DATASETS = [
#     DatasetEntry(
#         kind="robot",
#         path= os.path.join(base_pth, "ROBOT_PICK_PLACE_LEFT"),
#         weight=0.25,
#         task ="vertical_pick_place",
#         stride=2,
#         overlay=False,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None
#     ),

#     DatasetEntry(
#         kind="robot",
#         path=os.path.join(base_pth, "ROBOT_PICK_PLACE_RIGHT"),
#         weight=0.25,
#         task ="vertical_pick_place",
#         stride=2,
#         overlay=False,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None
#     ),


#     DatasetEntry(
#         kind="robot",
#         path=os.path.join(base_pth, "ROBOT_PICK_EXTRA_RIGHT_1031"),
#         weight=0.25,
#         task ="vertical_pick_place",
#         stride=2,
#         overlay=False,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None
#     ),


#     DatasetEntry(
#         kind="robot",
#         path=os.path.join(base_pth, "ROBOT_PICK_PLACE_INTER_1030"),
#         weight=0.25,
#         task ="vertical_pick_place",
#         stride=2,
#         overlay=False,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None
#     ),
# ]

###############################################################################
###############################################################################
###############################################################################
# BLOCK SORT TASK
###############################################################################
###############################################################################
###############################################################################

#####################################################
# Block plate sort dataset cotrain REVERSED (green left, yellow right) 11052025
######################################################
# TODO: NOTE: set accordingly, set False for ablation study
# overlay = True
# DATASETS = [
#     DatasetEntry(
#         kind="robot",
#         path="/iris/projects/humanoid/dataset/ROBOT_SORT_TL_1104",
#         weight=0.25,
#         task ="vertical_pick_place",
#         stride=2,
#         overlay=overlay,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None,
#         ee_to_hand_left_xyz = [0.01, -0.019, -0.04], # TODO: this is added for the new composition data collected 1110... instead of collect human data we collected robot data
#         ee_to_hand_right_xyz = [-0.01, -0.005, 0.015]
#     ),
#     DatasetEntry(
#         kind="robot",
#         path="/iris/projects/humanoid/dataset/ROBOT_SORT_TR_1104",
#         weight=0.25,
#         task ="vertical_pick_place",
#         stride=2,
#         overlay=overlay,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None,
#         ee_to_hand_left_xyz = [0.01, -0.019, -0.04], # TODO: this is added for the new composition data collected 1110... instead of collect human data we collected robot data
#         ee_to_hand_right_xyz = [-0.01, -0.005, 0.015]
#     ),
#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_SORT_TL_1104_COPY",
#         weight=0.125,
#         overlay=overlay,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None
#     ),

#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_SORT_TR_1104_COPY",
#         weight=0.125,
#         overlay=overlay,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None
#     ),

#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_SORT_IL_REV_1105", # TODO: NOTE: this is the reversed dataset
#         # path="/iris/projects/humanoid/dataset/HUMAN_SORT_IL_1104_COPY",
#         weight=0.125,
#         overlay=overlay,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None
#     ),

#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_SORT_IR_REV_1105", #TODO: NOTE: this is the reversed dataset
#         # path="/iris/projects/humanoid/dataset/HUMAN_SORT_IR_1104_COPY",
#         weight=0.125,
#         overlay=overlay,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None
#     ),
# ]

#####################################################
# Block plate sort dataset cotrain (yellow left, green right) 11052025
######################################################
# DATASETS = [
#     # DatasetEntry(
#     #     kind="egodex",
#     #     path="/iris/projects/humanoid/dataset/ego_dex",
#     #     weight=0.0,  # no training yet
#     # ),
#     DatasetEntry(
#         kind="robot",
#         path="/iris/projects/humanoid/dataset/ROBOT_SORT_TL_1104",
#         weight=0.25,
#         task ="vertical_pick_place",
#         stride=2,
#         overlay=True,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None
#     ),
#     DatasetEntry(
#         kind="robot",
#         path="/iris/projects/humanoid/dataset/ROBOT_SORT_TR_1104",
#         weight=0.25,
#         task ="vertical_pick_place",
#         stride=2,
#         overlay=True,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None
#     ),
#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_SORT_TL_1104_COPY",
#         weight=0.125,
#         overlay=True,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None
#     ),
    
#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_SORT_TR_1104_COPY",
#         weight=0.125,
#         overlay=True,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None
#     ),

#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_SORT_IL_1104_COPY",
#         weight=0.125,
#         overlay=True,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None
#     ),

#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_SORT_IR_1104_COPY",
#         weight=0.125,
#         overlay=True,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None
#     ),
# ]

#####################################################
# Block plate sort dataset baseline (robot data only) 11042025
######################################################
# DATASETS = [
#     DatasetEntry(
#         kind="robot",
#         path="/iris/projects/humanoid/dataset/ROBOT_SORT_TL_1104",
#         weight=0.5,
#         task ="vertical_pick_place",
#         stride=2,
#         overlay=True,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None
#     ),
#     DatasetEntry(
#         kind="robot",
#         path="/iris/projects/humanoid/dataset/ROBOT_SORT_TR_1104",
#         weight=0.5,
#         task ="vertical_pick_place",
#         stride=2,
#         overlay=True,
#         mask_wrist=True,
#         apply_custom_norm=False,
#         norm_stats_path = None
#     ),
# ]
