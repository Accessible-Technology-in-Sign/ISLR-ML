import mediapipe as mp
import cv2
import os
import numpy as np

__hands = mp.solutions.hands.Hands(
    static_image_mode = os.environ.get("STATIC_IMAGE_MODE", False),
    max_num_hands = os.environ.get("MAX_NUM_HANDS", 1),
    min_detection_confidence=os.environ.get("MIN_DETECTION_CONFIDENCE", 0.5),
    min_tracking_confidence=os.environ.get("MIN_TRACKING_CONFIDENCE", 0.5)
)

def run(frame, colorspace_convert=cv2.COLOR_BGR2RGB, **kwargs):
    if 'flipH' in kwargs and kwargs['flipH']:
        cv2.flip(frame, 1)
    if 'flipV' in kwargs and kwargs['flipV']:
        cv2.flip(frame, 0)

    if colorspace_convert:
        frame = cv2.cvtColor(frame, colorspace_convert)
    return __hands.process(frame)

# TODO - normalize handedness
def post_process(results, keypoints=mp.solutions.hands.HandLandmark, world=True, coords="xyz", normalize_handedness = "left", **kwargs):
    lm_list = results.multi_hand_landmarks
    if world:
        lm_list = results.multi_hand_world_landmarks
    if lm_list:
        return np.array([[[
            1 - getattr(lm.landmark[keypoint], coord)
                if results.multi_handedness[idx].classification[0].label.lower() == "left" and coord == "x" and not world
            else getattr(lm.landmark[keypoint], coord)
            for coord in coords
            ] for keypoint in keypoints] for idx, lm in enumerate(lm_list)])
    else:
        return np.zeros((1, 21, len(coords)))


