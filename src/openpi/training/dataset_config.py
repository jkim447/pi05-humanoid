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


base_pth = "/iris/projects/humanoid/dataset/"

# TODO: make sure ONLY your desired datasets are uncommented!
# TODO: ensure the ratio of each data!
# TODO: scroll to the bottom to check, since it's a long file!

###############################################################################
###############################################################################
###############################################################################
# COMPOSITION -- Box pull + pick place
###############################################################################
###############################################################################
###############################################################################

# TODO: NOTE: I'm using the same langauge instruction for everything for this task!
DATASETS = [

    # below is composition data
    DatasetEntry(
        kind="human",
        path="/iris/projects/humanoid/dataset/HUMAN_BOX_PLACE_COMBO_1105",
        weight=0.2,
        overlay=True,
        task = "vertical_pick_place",
        apply_custom_norm=False,
        norm_stats_path=None,
        overlay_both=True, # TODO: the option to overlay both hands is set!
        both_actions=True,# TODO: this is bimanual action data so return both valid left / right actions!
    ),

    # below is box pull data
    DatasetEntry(
        kind="robot",
        path="/iris/projects/humanoid/dataset/ROBOT_PULL_BOX_1105",
        weight=0.14,
        task ="vertical_pick_place", # TODO: set correct language instructions across
        stride=2,
        overlay=True,
        mask_wrist=True,
        apply_custom_norm=False,
        norm_stats_path = None
    ),

    DatasetEntry(
        kind="human",
        path="/iris/projects/humanoid/dataset/HUMAN_PULL_BOX_1105",
        weight=0.14,
        overlay=True,
        task = "vertical_pick_place",
        apply_custom_norm=False,
        norm_stats_path=None,
        overlay_both=True, # TODO: the option to overlay both hands is set!
        both_actions=True, # TODO: this is bimanual action data so return both valid left / right actions!
    ),

    # below is sort cotrain data
    DatasetEntry(
        kind="robot",
        path="/iris/projects/humanoid/dataset/ROBOT_SORT_TL_1104",
        weight=0.13,
        task ="vertical_pick_place",
        stride=2,
        overlay=True,
        mask_wrist=True,
        apply_custom_norm=False,
        norm_stats_path = None
    ),
    DatasetEntry(
        kind="robot",
        path="/iris/projects/humanoid/dataset/ROBOT_SORT_TR_1104",
        weight=0.13,
        task ="vertical_pick_place",
        stride=2,
        overlay=True,
        mask_wrist=True,
        apply_custom_norm=False,
        norm_stats_path = None
    ),
    DatasetEntry(
        kind="human",
        path="/iris/projects/humanoid/dataset/HUMAN_SORT_TL_1104_COPY",
        weight=0.13,
        overlay=True,
        task = "vertical_pick_place",
        apply_custom_norm=False,
        norm_stats_path=None
    ),
    
    DatasetEntry(
        kind="human",
        path="/iris/projects/humanoid/dataset/HUMAN_SORT_TR_1104_COPY",
        weight=0.13,
        overlay=True,
        task = "vertical_pick_place",
        apply_custom_norm=False,
        norm_stats_path=None
    ),
]



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
# DATASETS = [
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
#         path="/iris/projects/humanoid/dataset/HUMAN_SORT_IL_REV_1105",
#         weight=0.125,
#         overlay=True,
#         task = "vertical_pick_place",
#         apply_custom_norm=False,
#         norm_stats_path=None
#     ),

#     DatasetEntry(
#         kind="human",
#         path="/iris/projects/humanoid/dataset/HUMAN_SORT_IR_REV_1105",
#         weight=0.125,
#         overlay=True,
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
