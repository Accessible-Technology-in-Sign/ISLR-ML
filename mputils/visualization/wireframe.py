#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: wireframe.py
Author: Rohit Sridhar
Date: -
Last Modified: Sun 29 Mar 2026 02:35:21 PM EDT
Version: X.X

Description:
    Contains code to generate wireframe videos, given a parquet file 
    as input. Filters by paricipant ID, filter video dir (takes the 
    intersection with metadata videos) and can sample N videos.
"""

import sys
import math
import random
import logging

import pandas as pd
from pyarrow import parquet as pq
from pathlib import Path
from tqdm import tqdm

import re
import os
import shutil
import argparse

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np

from args import *
from utils import (
    setup_logger,
    get_tqdm_filter_desc,
    get_video_path,
    FPS,
)

# TODO Add proper relative import later
# import sys
# sys.path.append('../scripts')

# from utils import *

WIREFRAME_PATH = Path().resolve()
LOGS_PATH = Path().resolve()
# LOGS_PATH = Path("logs/wireframe").resolve()

PARQUET_FEATURE_LIST = [
    'sequence_id',
    'frame',
    *[
        f'{coord}_{hand}_{i}'
        for hand in ['right']
        for coord in ['x', 'y', 'z']
        for i in range(21)
    ],
]
PARQUET_RH_FEATURES = [i for i in range(0, 42)]

# global args
def set_global_paths():
    global WIREFRAME_PATH
    global LOGS_PATH
    
    WIREFRAME_PATH = WIREFRAME_PATH / "videos"/ "wireframe"
    LOGS_PATH = LOGS_PATH / "logs" / "wireframe"
    
    if args.interpolate is not None:
        WIREFRAME_PATH /= f"interp{args.interpolate}"
        LOGS_PATH /= f"interp{args.interpolate}"

# Video Path: videos/1967755728/f418
# /data/parquet/asl-fingerspelling/train_landmarks/1967755728.parquet
def load_parquet(filename) -> pd.DataFrame:
    logging.info(f"Parquet file: {filename}")
    parquet_array = pq.read_table(filename, columns=PARQUET_FEATURE_LIST, memory_map=True).to_pandas()
    if args.debug:
        dbg_sample = parquet_array.sample(n=50, random_state=args.seed).index
        parquet_array = parquet_array.loc[dbg_sample]
    return parquet_array

# render a wireframe video file using landmark data and save it to disk
def render(data, filename, pt_id, wireframe=True):
    x_features = [i for i in range(0, 21)]
    y_features = [i for i in range(21, 42)]
    # z_features = [i for i in range(42, 63)]
    
    last_good_frame = 0
    repeated = data[:]
    
    for i, frame in enumerate(data):
        if not math.isnan(frame[0]):
            repeated[i] = data[i]
            last_good_frame = i
        else:
            repeated[i] = data[last_good_frame]

    repeated = [frame for frame in repeated if not math.isnan(frame[0])]
    
    xs = [frame[x_features] for frame in repeated]
    ys = [1 - frame[y_features] for frame in repeated]
    # zs = [frame[z_features] for frame in repeated]


    logging.info(f'{len(xs)=}, {len(ys)=}')
    if len(xs) == 0:
        logging.info(f'{filename} empty data')
        return

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)
    sct, = ax.plot([], [], "o", markersize=2)
    # sct, = ax.plot([], [], [], "o", markersize=2)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.set_zlabel('z')

    # ax.invert_zaxis()

    lines = list()

    def update(ifrm, xa, ya):
    # def update(ifrm, xa, ya, za):
        nonlocal lines
        logging.debug(f'update({ifrm=})')
        # ax.view_init(elev=(20 + ifrm * 0.25), azim=(90 + ifrm * 0.05))
        sct.set_data(xa[ifrm], ya[ifrm])
        # sct.set_3d_properties(za[ifrm])

        if wireframe:
            BLACK = 'black'
            RED = 'red'
            BLUE = 'blue'
            YELLOW = 'yellow'
            ORANGE = 'orange'
            GREEN = 'green'

            lines_between = [
                [0, 1, BLACK], [1, 2, RED], [2, 3, RED], [3, 4, RED],
                [1, 5, BLACK], [5, 6, BLUE], [6, 7, BLUE], [7, 8, BLUE],
                [5, 9, BLACK], [9, 10, YELLOW], [10, 11, YELLOW], [11, 12, YELLOW],
                [9, 13, BLACK], [13, 14, ORANGE], [14, 15, ORANGE], [15, 16, ORANGE],
                [13, 17, BLACK], [0, 17, BLACK], [17, 18, GREEN], [18, 19, GREEN], [19, 20, GREEN]
            ]

            for line in lines:
                line.remove()

            lines = list()
            
            for from_pt, to_pt, color in lines_between:
                if np.isnan(xa[ifrm][from_pt]) or np.isnan(xa[ifrm][to_pt]):
                    continue
                
                # logging.info(len(ax.plot([xa[ifrm][from_pt], xa[ifrm][to_pt]], [ya[ifrm][from_pt], ya[ifrm][to_pt]], color=color)))
                lines.append(
                    ax.plot([xa[ifrm][from_pt], xa[ifrm][to_pt]],
                            [ya[ifrm][from_pt], ya[ifrm][to_pt]],
                            color=color)[0]
                )
                    # ax.plot(xs=[xa[ifrm][from_pt], xa[ifrm][to_pt]],
                    #         ys=[ya[ifrm][from_pt], ya[ifrm][to_pt]],
                    #         zs=[za[ifrm][from_pt], za[ifrm][to_pt]],
                    #         color=color)[0]

        # print(f'{ifrm=}, {xa[ifrm]=}, {ya[ifrm]=}, {za[ifrm]=}')

    zoom = 1.0
    ax.set_xlim(-zoom / 10, zoom)
    ax.set_ylim(-zoom / 10, zoom)
    # ax.set_zlim(-zoom / 10, zoom)
    # ani = animation.FuncAnimation(fig, update, len(repeated), fargs=(xs, ys, zs), interval=1000 / FPS)
    ani = animation.FuncAnimation(fig, update, len(repeated), fargs=(xs, ys), interval=1000 / FPS)
    
    video_path = get_video_path(
        args.parquet_file,
        WIREFRAME_PATH,
        filter_video_dir=args.filter_video_dir,
        filter_pt_id=args.filter_pt_id,
    )
    video_file = video_path / f"{filename}.mp4"

    ani.save(video_file, writer='ffmpeg', fps=FPS)
    logging.info(f'Made {video_file}')
    plt.close(fig)

# filter metadata by video dir
def filter_by_video_dir(metadata):
    filter_videos = list(args.filter_video_dir.glob("*.mp4"))
    filter_videos = [video.name for video in filter_videos]
    metadata = metadata.loc[metadata.clipFilename.isin(filter_videos)]

    logging.info(f"Sequences after filtering by video dir: {metadata.shape[0]}")
    return metadata

# filter metadata by pt id
def filter_by_pt_id(metadata):
    metadata = metadata.astype({"participant_id" : str})
    metadata = metadata.loc[metadata.participant_id == args.filter_pt_id]

    logging.info(f"Sequences after filtering by participant: {metadata.shape[0]}")
    return metadata

def interpolate_data(data):
    def interpolate(row, interpolate_val=0):
        X = row.x_right_0 == 0.0
        if X.all():
            return row
        X = X.to_list()
        
        start = X.index(False)
        end = len(X) - X[::-1].index(False)
        if start == end:
            return row
        
        row = row.reset_index(drop=True)
        row_nan = row.iloc[start:end].replace(0.0, np.nan)
        # row_nan = row.iloc[start:end]
        row_nan = row_nan.interpolate()

        if interpolate_val == 0:
            return pd.concat([row.iloc[:start], row_nan, row.iloc[end:]])

        fwd_row = row_nan[1:].reset_index(drop=True)
        bck_row = row_nan[:-1].reset_index(drop=True)
        row_sum = fwd_row + bck_row

        interp_row = (row_sum / 2)
        subset_row = row_nan.reset_index(drop=True)
        
        interp_row["order"] = list(range(1, interp_row.shape[0] * 2, 2))
        subset_row["order"] = list(range(0, subset_row.shape[0] * 2, 2))
        interp_row = interp_row.set_index("order")
        subset_row = subset_row.set_index("order")
        
        new_row = pd.concat([interp_row, subset_row])
        new_row = new_row.sort_index().reset_index().drop("order", axis=1)
        new_row = pd.concat([row.iloc[:start], new_row, row.iloc[end:]])

        return new_row
    
    data = data.drop(columns=["frame"]).fillna(0.0)
    data = data.groupby("sequence_id").apply(
        interpolate,
        interpolate_val=args.interpolate,
    )
    data = data.reset_index().rename(columns={"level_1":"frame"}).set_index("sequence_id")
    return data

# generate videos from filtered sequence ids in a loop.
def generate_videos():
    pq_data = load_parquet(args.parquet_file)
    if args.interpolate is not None:
        pq_data = interpolate_data(pq_data)
    
    logging.debug(pq_data)
    sys.exit(1)

    metadata_file = (
        args.parquet_file.parents[1] /
        "metadata" /
        args.parquet_file.name
    ).with_suffix(".csv")
    metadata = pd.read_csv(metadata_file).set_index("sequence_id")
    seq_ids = metadata.index.values.tolist()

    if args.filter_video_dir is not None:
        metadata = filter_by_video_dir(metadata)

    if args.filter_pt_id is not None:
        metadata = filter_by_pt_id(metadata)

    seq_ids = metadata.index.values.tolist()
    seq_ids = random.sample(seq_ids, min(len(seq_ids), args.sample_size))

    tqdm_desc = get_tqdm_filter_desc(
        args.parquet_file.stem,
        filter_video_dir=args.filter_video_dir,
        filter_pt_id=args.filter_pt_id,
    )
    for seq_id in tqdm(
        seq_ids,
        desc=tqdm_desc,
        position=args.bar_position,
        disable=args.std_out,
    ):
        row = metadata.loc[seq_id]
        phrase = row.phrase
        
        frames = pq_data[pq_data.index == seq_id]
        frames = frames.iloc[:,1:].to_numpy()
        
        phrase_path = re.sub(r'[:/\\ ]', '_', phrase)
        render(
            frames[:, PARQUET_RH_FEATURES],
            f"{phrase_path}_{str(seq_id)}_rh_wire",
            row.participant_id,
            True
        )


if __name__ == '__main__':
    args = parse_args()
    check_args(args)

    random.seed(args.seed)
    set_global_paths()

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

    generate_videos()

