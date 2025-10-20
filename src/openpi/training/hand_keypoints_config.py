# hand_keypoints_config.py
from typing import Dict, Tuple, List

LEFT_FINGERS: List[List[str]] = [
    ["leftThumbKnuckle","leftThumbIntermediateBase","leftThumbIntermediateTip","leftThumbTip"],
    ["leftIndexFingerKnuckle","leftIndexFingerIntermediateBase","leftIndexFingerIntermediateTip","leftIndexFingerTip"],
    ["leftMiddleFingerKnuckle","leftMiddleFingerIntermediateBase","leftMiddleFingerIntermediateTip","leftMiddleFingerTip"],
    ["leftRingFingerKnuckle","leftRingFingerIntermediateBase","leftRingFingerIntermediateTip","leftRingFingerTip"],
    ["leftLittleFingerKnuckle","leftLittleFingerIntermediateBase","leftLittleFingerIntermediateTip","leftLittleFingerTip"],
]

RIGHT_FINGERS: List[List[str]] = [
    ["rightThumbKnuckle","rightThumbIntermediateBase","rightThumbIntermediateTip","rightThumbTip"],
    ["rightIndexFingerKnuckle","rightIndexFingerIntermediateBase","rightIndexFingerIntermediateTip","rightIndexFingerTip"],
    ["rightMiddleFingerKnuckle","rightMiddleFingerIntermediateBase","rightMiddleFingerIntermediateTip","rightMiddleFingerTip"],
    ["rightRingFingerKnuckle","rightRingFingerIntermediateBase","rightRingFingerIntermediateTip","rightRingFingerTip"],
    ["rightLittleFingerKnuckle","rightLittleFingerIntermediateBase","rightLittleFingerIntermediateTip","rightLittleFingerTip"],
]

# Distinct palettes (left = warm hues, right = cool hues)
_LEFT_BASE = [
    (60, 70, 255),   # red
    (60, 180, 255),  # orange
    (80, 255, 180),  # green
    (180, 255, 100), # yellow-green
    (255, 150, 80),  # light blue
]
_RIGHT_BASE = [
    (255, 100, 60),  # blue
    (255, 80, 180),  # magenta
    (255, 180, 255), # pink
    (180, 80, 255),  # violet
    (100, 255, 255), # cyan
]

def build_joint_color_map() -> Dict[str, Tuple[int,int,int]]:
    cmap: Dict[str, Tuple[int,int,int]] = {}
    cmap["leftHand"] = (150, 150, 150)
    cmap["rightHand"] = (150, 150, 150)

    # Left hand
    for fidx, chain in enumerate(LEFT_FINGERS):
        base = _LEFT_BASE[fidx]
        for step, name in enumerate(chain):
            cmap[name] = base

    # Right hand
    for fidx, chain in enumerate(RIGHT_FINGERS):
        base = _RIGHT_BASE[fidx]
        for step, name in enumerate(chain):
            cmap[name] = base

    return cmap

JOINT_COLOR_BGR: Dict[str, Tuple[int,int,int]] = build_joint_color_map()
