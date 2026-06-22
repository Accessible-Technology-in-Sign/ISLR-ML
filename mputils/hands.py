from pathlib import Path

import mediapipe as mp
import cv2
import os
import numpy as np

__hands = mp.solutions.hands.Hands(
    static_image_mode = os.environ.get("STATIC_IMAGE_MODE", False),
    max_num_hands = os.environ.get("MAX_NUM_HANDS", 1),
    min_detection_confidence=float(os.environ.get("MIN_DETECTION_CONFIDENCE", 0.5)),
    min_tracking_confidence=float(os.environ.get("MIN_TRACKING_CONFIDENCE", 0.5))
)

def run(frame, colorspace_convert=cv2.COLOR_BGR2RGB, **kwargs):
    if 'flipH' in kwargs and kwargs['flipH']:
        cv2.flip(frame, 1)
    if 'flipV' in kwargs and kwargs['flipV']:
        cv2.flip(frame, 0)

    if 'frame_name' in kwargs and isinstance(kwargs['frame_name'], Path):
        cv2.imwrite(str(kwargs['frame_name']), frame)

    if colorspace_convert:
        frame = cv2.cvtColor(frame, colorspace_convert)
    return __hands.process(frame)

# TODO - normalize handedness
def post_process(results, keypoints=mp.solutions.hands.HandLandmark, world=True, coords="xyz", normalize_handedness = "left", **kwargs):
    lm_list = results.multi_hand_landmarks
    if world:
        lm_list = results.multi_hand_world_landmarks
    if lm_list:
        extracted_landmarks = []
        for idx, lm in enumerate(lm_list):
            left_handed = results.multi_handedness[idx].classification[0].label.lower() == "left"
            # print(f"Left Handed: {left_handed}")

            lm_values = []
            for coord in coords:
                # print(f"Coord: {coord}")
                for keypoint in keypoints:
                    # print(f"Keypoint: {keypoint}")
                    # print(f"Original LM: {lm.landmark[keypoint]}")
                    if left_handed and coord == "x":
                        # print(f"lm val: {1 - getattr(lm.landmark[keypoint], coord)}")
                        lm_values.append(1 - getattr(lm.landmark[keypoint], coord))
                    else:
                        # print(f"lm val: {getattr(lm.landmark[keypoint], coord)}")
                        lm_values.append(getattr(lm.landmark[keypoint], coord))
                    # print()
            extracted_landmarks.append(lm_values)
        extracted_landmarks = np.array(extracted_landmarks).squeeze()
        return extracted_landmarks, left_handed
    else:
        return np.zeros((63)), None
        # return np.zeros((1, 21, len(coords)))


