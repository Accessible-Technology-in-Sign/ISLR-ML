#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: overlay.py
Author: Rohit Sridhar
Date: -
Last Modified: Sun 29 Mar 2026 02:35:02 PM EDT
Version: X.X

Description:
    Contains code to generate mediapipe overlays, using the source video.
    Can filter by video dir (to only matching videos in smaller dir) and/or 
    participant ID. Can also sample N videos.
"""

import cv2
import sys
import logging
import ffmpeg
import time
import argparse
import os
import glob
import json
import random

import mediapipe as mp
import pandas as pd

from args import *
from utils import (
    setup_logger,
    get_tqdm_filter_desc,
    get_video_path,
    OVERLAY_PATH,
)
from tqdm import tqdm
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark, NormalizedLandmarkList
from multiprocessing import Pool
from pathlib import Path

global args

LOGS_PATH = Path("logs/overlay").resolve()

# def parse_args():
#     """ This is executed when run from the command line """
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     
#     parser.add_argument("-pt", "--filter_pt_id", type=str, default=None, help="participant filtering. optional.")
#     parser.add_argument("-fv", "--filter_video_dir", default=None, type=Path, help="Filter sequences to only videos in dir.")
#     parser.add_argument("-ss", "--sample_size", type=int, default=100, help="sample size.")
#     parser.add_argument("-vs", "--video_src", type=Path, required=True, help="directory with videos.")
#     parser.add_argument("-pf", "--parquet_file", type=Path, required=True, help="parquet file for processing.")
#     parser.add_argument("-sd", "--seed", type=int, default=7987, help="seed for random sampling.")
#     parser.add_argument("-bp", "--bar_position", type=int, default=0, help="Bar position for tqdm.")
#     parser.add_argument("-std", "--std_out", action="store_true", help="output to standard out (tqdm will be disabled).")
#     parser.add_argument("-dbg", "--debug", action="store_true", help="output to standard out (tqdm will be disabled).")
#     
#     args = parser.parse_args()
#     return args
# 
# set up the logger
# def setup_logger():
#     log_filename = f"sd{args.seed}_n{args.sample_size}"
#     if args.pt_id is not None:
#         log_filename += f"_pt-{args.pt_id}"
# 
#     log_file = (
#         LOGS_PATH /
#         args.parquet_file.stem /
#         log_filename
#     ).with_suffix(".log")
# 
#     filemode='w'
#     # if log_file.exists():
#     #     filemode='a'
# 
#     log_file.parent.mkdir(parents=True, exist_ok=True)
#     if args.std_out:
#         logging.basicConfig(
#             stream=sys.stdout,
#             level=logging.DEBUG if args.debug else logging.INFO,
#             format="%(asctime)s - %(levelname)s - %(message)s",
#             datefmt="%Y-%m-%d | %H:%M:%S"
#         )
#     else:
#         logging.basicConfig(
#             filename=log_file,
#             filemode=filemode,
#             level=logging.DEBUG if args.debug else logging.INFO,
#             format="%(asctime)s - %(levelname)s - %(message)s",
#             datefmt="%Y-%m-%d | %H:%M:%S"
#         )

def format_hand_landmarks(frame_i, landmarks):
    frame_lm = landmarks.loc[landmarks.frame == frame_i].drop("frame", axis=1)
    if frame_lm.isna().all(axis=1).item():
        return None
    else:
        frame_data = frame_lm.iloc[0].to_list()
        right_hand_landmarks = NormalizedLandmarkList(
          landmark = [NormalizedLandmark(x=frame_data[x], y=frame_data[x+21], z=frame_data[x+21*2]) for x in range(21)]
        )
        return right_hand_landmarks

def draw_mediapipe_landmarks(video_details):
    video_filepath = video_details["video_filepath"]
    new_video_filepath = video_details["new_video_filepath"]
    hand_landmarks = video_details["hand_landmarks"]

    # use prefix for logging in this function (since multithreaded)
    log_prefix = new_video_filepath.stem

    logging.info(f"[{log_prefix}] Original Video Filepath: {video_filepath}")
    logging.info(f"[{log_prefix}] New Video Filepath: {new_video_filepath}")
    
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    
    # For video input:
    cap = cv2.VideoCapture(video_filepath)
    result = cv2.VideoWriter(filename=new_video_filepath,
                           fourcc=cv2.VideoWriter.fourcc(*"mp4v"),
                           fps=cap.get(cv2.CAP_PROP_FPS), frameSize=(int(cap.get(3)), int(cap.get(4))))
    logging.debug(f"[{log_prefix}] FPS Support: {result.get(cv2.CAP_PROP_FPS)}")
    logging.debug(f"[{log_prefix}] Input Vid Orientation: {cap.get(cv2.CAP_PROP_ORIENTATION_META)}")

    orientation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    start = time.time()
    num_frames = 0
    pose_null = 0
    printed = False

    video_metadata = ffmpeg.probe(video_filepath)['streams'][0]['tags']
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            logging.info(f"[{log_prefix}] Ignoring empty camera frame.")
            # If loading a video, use "break" instead of "continue".
            break
        original_image_shape = image.shape # used for debugging
        
        # if "rotate" in metadata:
        #     image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #     if not printed:
        #         logging.debug(f"[{log_prefix}] Frame Shape After Rotation: {image.shape}")
        #     # image = cv2.resize(image, (1080, 1920))
        
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        
        right_hand_landmarks = format_hand_landmarks(
            num_frames,
            hand_landmarks
        )

        mp_drawing.draw_landmarks(
            image, right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # mp_drawing.draw_landmarks(
        #     image, left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # mp_drawing.draw_landmarks(
        #     image, pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        # if pose_landmarks is None:
        #     pose_null += 1
        
        #define the screen resulation
        # screen_res = 1280, 720
        # scale_width = screen_res[0] / image.shape[1]
        # scale_height = screen_res[1] / image.shape[0]
        # scale = min(scale_width, scale_height)
        # #resized window width and height
        # window_width = int(image.shape[1] * scale)
        # window_height = int(image.shape[0] * scale)
        #cv2.WINDOW_NORMAL makes the output window resizealbe
        window_width = int(cap.get(3))
        window_height = int(cap.get(4))
        image = cv2.flip(image, 1)

        if not printed:
            logging.debug(f"[{log_prefix}] VideoWriter Expected Shape: [{window_height}, {window_width}]")
            logging.debug(f"[{log_prefix}] Original Frame Shape: {original_image_shape}")
            logging.debug(f"[{log_prefix}] Final Frame Shape: {image.shape}")
            printed = True

        result.write(image)
        # if show_overlay:
        #     cv2.imshow("Overlay", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        num_frames += 1
    
    end = time.time() - start
    logging.info(f"[{log_prefix}] Number of times pose is none = " + str(pose_null))
    logging.info(f"[{log_prefix}] Time taken = " + str(end))
    logging.info(f"[{log_prefix}] Total frames = " + str(num_frames))
    logging.info(f"[{log_prefix}] Frames processed per second = " + str(num_frames/end))
    # holistic.close()
    cap.release()
    result.release()
    
    return new_video_filepath

def get_parquet():
    metadata_file = Path(str(args.parquet_file).replace("landmarks", "metadata").replace(".parquet",".csv"))

    pq_data = pd.read_parquet(args.parquet_file)
    metadata = pd.read_csv(metadata_file)

    return pq_data, metadata

def filter_by_pt_files(src_files):
    """
    filters videos by participant id.
    """
    processed_src_files = []
    for src_f in src_files:
        basename = src_f.name
        pt_id = basename.split('_')[0]
        
        if pt_id == args.pt_id:
            processed_src_files.append(src_f)

    # processed_src_files = random.sample(processed_src_files, min(args.sample_size, len(processed_src_files)))
    return processed_src_files

def filter_by_video_files(src_files):
    filter_video_files = args.filter_video_dir.glob("*.mp4")
    filter_video_files = set([video.name for video in filter_video_files])
    
    processed_src_files = []
    for src_f in src_files:
        if src_f.name in filter_video_files:
            processed_src_files.append(src_f)
    
    return processed_src_files

if __name__ == "__main__":
    args = parse_args()
    check_args(args)
    random.seed(args.seed)
    setup_logger(
        args.seed,
        args.sample_size,
        args.parquet_file,
        LOGS_PATH,
        filter_pt_id=args.filter_pt_id,
        filter_video_dir=args.filter_video_dir,
        std_out=args.std_out,
        debug=args.debug,
    )

    pq_data, metadata = get_parquet()
    available_videos = set(metadata.clipFilename.to_list())
    
    src_files = sorted(list(args.video_src.glob("*.mp4")))
    src_files = [src_f for src_f in src_files if src_f.name in available_videos]
    logging.debug(f"Src Files before Filtering on PT/Video Dir{len(src_files)=}")
    if args.filter_pt_id is not None:
        src_files = filter_by_pt_files(src_files)

    if args.filter_video_dir is not None:
        src_files = filter_by_video_files(src_files)
        
    logging.debug(f"Src Files after Filtering on PT/Video Dir{len(src_files)=}")
    src_files = random.sample(src_files, min(args.sample_size, len(src_files)))
    
    mapped = []
    seen_phrases = set() 
    
    for src_f in src_files:
        filename = src_f.name

        metadata_row = metadata.loc[metadata["clipFilename"] == filename]
        hand_landmarks = pq_data.loc[metadata_row.sequence_id]

        phrase = metadata_row["phrase"].item().replace(' ','_') 
        seq_id = str(metadata_row["sequence_id"].item())
        # pt_id = filename.split('_')[0]
        
        dest = get_video_path(
            args.parquet_file,
            OVERLAY_PATH,
            filter_video_dir=args.filter_video_dir,
            filter_pt_id=args.filter_pt_id,
        )
        new_vidname = '_'.join([seq_id, phrase + ".mp4"])
        dest_f = dest / new_vidname
        logging.debug(f"Destination File just after get_video_path call: {dest_f}")

        mapped.append({
            "video_filepath": src_f,
            "new_video_filepath": dest_f,
            "hand_landmarks": hand_landmarks
        })
    
    
    with Pool(processes=(os.cpu_count() // 4)) as pool:
        tqdm_desc = get_tqdm_filter_desc(
            args.parquet_file.stem,
            filter_video_dir=args.filter_video_dir,
            filter_pt_id=args.filter_pt_id,
        )
        results = list(tqdm(
            pool.imap(draw_mediapipe_landmarks, mapped),
            desc=tqdm_desc,
            position=args.bar_position,
            disable=args.std_out,
            total=len(mapped),
        ))
    logging.info(results)
    
