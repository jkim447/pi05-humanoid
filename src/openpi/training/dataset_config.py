# openpi/training/dataset_config.py
from typing import NamedTuple, Literal

class DatasetEntry(NamedTuple):
    kind: Literal["egodex", "robot", "human"]
    path: str
    weight: float              # target sampling probability
    task: str = "vertical_pick_place"
    stride: int = 2
    overlay: bool = True
    mask_wrist: bool = True
    apply_custom_norm: bool = True

# TODO: check the parameters of the datasets you're using!
DATASETS = [
    # DatasetEntry(
    #     kind="egodex",
    #     path="/iris/projects/humanoid/dataset/ego_dex",
    #     weight=0.0,  # no training yet
    # ),
    DatasetEntry(
        kind="robot",
        path="/iris/projects/humanoid/dataset/ROBOT_SORT_TL_1104",
        weight=0.5,
        task ="vertical_pick_place",
        stride=2,
        overlay=True,
        mask_wrist=True,
        apply_custom_norm=True,
    ),
    DatasetEntry(
        kind="robot",
        path="/iris/projects/humanoid/dataset/ROBOT_SORT_TR_1104",
        weight=0.5,
        task ="vertical_pick_place",
        stride=2,
        overlay=True,
        mask_wrist=True,
        apply_custom_norm=True
    ),
    # DatasetEntry(
    #     kind="human",
    #     path="/iris/projects/humanoid/dataset/HUMAN_SORT_TL_1101",
    #     weight=0.15,
    # ),
    # DatasetEntry(
    #     kind="human",
    #     path="/iris/projects/humanoid/dataset/HUMAN_SORT_TR_1101",
    #     weight=0.15,
    # ),
]
