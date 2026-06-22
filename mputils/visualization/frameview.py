#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: frameview.py
Author: Rohit Sridhar
Date: -
Last Modified: Sat 28 Mar 2026 02:21:16 AM EDT
Version: X.X

Description:
    Contained code to generate frameview, given a parquet
    file as input.
"""

import os
import argparse

import pandas as pd
from pyarrow import parquet as pq
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from utils import FRAMEVIEW_PATH

# INPUT_FOLDER = f'/data/deep_learning/ISLR-ML/mputils/out/landmarks'
FRAMEVIEW_PATH = Path("frameview").resolve()

PARQUET_FEATURE_LIST = [
    *[
        f'{coord}_{hand}_{i}'
        for hand in ['left_hand', 'right_hand']
        for coord in ['x', 'y', 'z']
        for i in range(21)
    ],
    'frame',
    'sequence_id'
]

# NOTE: Feature list is `[xyz]_right_[0-20]`, not `_right_hand_`
PARQUET_RH_FEATURE_LIST = [
    *[
        f'{coord}_right_{i}'
        for coord in ['x', 'y', 'z']
        for i in range(21)
    ],
    'frame',
    'sequence_id'
]

PARQUET_LH_FEATURES = [i for i in range(0, 63)]
PARQUET_RH_FEATURES = [i for i in range(63, 126)]

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-pf",
        "--parquet_file",
        type=Path,
        help="Parquet file with data to visualize."
    )
    
    return parser.parse_args()

def load_parquet(filename, rh_only=False) -> pd.DataFrame:
    parquet_array = pq.read_table(
        filename,
        columns=PARQUET_RH_FEATURE_LIST if rh_only else PARQUET_FEATURE_LIST,
        memory_map=True
    ).to_pandas()
    # print(f'Number of rows in {filename}: {len(parquet_array.index):,}')
    return parquet_array

def dqs(drops, skips, total):
    return 100 * ((drops/total) * 3 + (skips/total))

def dropped_frame_analysis(
    metadata,
    source_file,
    drop_threshold=5,
    rh_only=False
):
    metadata = pd.read_csv(metadata)

    file_list = [f'{i+1}' for i in range((metadata.shape[0] // 1000) + 1)]
    pq_data = load_parquet(source_file, rh_only=rh_only)

    output_dir = FRAMEVIEW_PATH / source_file.stem
    score_data = list()
    for i, file in enumerate(tqdm(file_list)):
        start = i * 1000
        end = min((i + 1) * 1000, metadata.shape[0])
        entries = metadata.iloc[start:end, :]

        plot_data = list()
        rh_features = PARQUET_LH_FEATURES if rh_only else PARQUET_RH_FEATURES
        ex_done, limit, total_ex = 0, -1, entries.shape[0]

        output_file = (output_dir / f"{file}_frames").with_suffix(".png")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        for index, row in entries.iterrows():
            seq_id = row['sequence_id']
            phrase = row['phrase']
            frames = pq_data[pq_data.index == seq_id]
            frames = frames.to_numpy()

            rh = np.sign(
                np.sum(np.isnan(frames[:, rh_features]), axis=1)
            )

            ex_data = rh

            if not rh_only:
                lh = np.sign(
                    np.sum(np.isnan(frames[:, PARQUET_LH_FEATURES]), axis=1)
                )

                if np.sum(lh) < np.sum(rh):
                    ex_data = lh

            total_frames = ex_data.shape[0]
            if total_frames < len(phrase) * 2:
                continue

            dropped_regions = np.split(ex_data,
                                       np.where(np.diff(ex_data) != 0)[0] + 1)
            data_sizes = [len(reg) for reg in dropped_regions if reg[0] == 0]
            drop_sizes = [len(reg) for reg in dropped_regions if reg[0] == 1]

            invalid = np.sum(ex_data)
            drops = sum([drop for drop in drop_sizes if drop >= drop_threshold])
            skips = invalid - drops

            score = dqs(drops, skips, total_frames)

            # Print examples with lots of missing data
            # if score > 100:
            #    print(f'For {phrase=}, {drop_sizes=}')

            paint_regions = list()
            start = 0
            for data, drop in zip(data_sizes, drop_sizes):
                start += data
                start_fp = start / total_frames
                region_len = drop / total_frames
                paint_regions.append((start_fp, start_fp + region_len))
                start += drop

            plot_data.append((score, phrase, paint_regions, seq_id))

            ex_done += 1
            if limit != -1 and ex_done >= limit:
                break

        plot_data.sort(key=lambda x: x[3])

        # plt.figure(figsize=(10, len(plot_data) * 0.08), dpi=80)
        plt.figure(figsize=(10, len(plot_data) * 0.5), dpi=80)
        labels = []
        ticks = []
        for i, (score, phrase, regions, seq_id) in enumerate(plot_data):
            labels.append(seq_id)
            ticks.append(i)
            if not regions:
                plt.hlines(i, 0, 1, 'lightgray', lw=2)
                continue

            if regions[0][0] > 0:
                plt.hlines(i, 0, regions[0][0], 'lightgray', lw=2)

            for j, (start, end) in enumerate(regions):
                plt.hlines(i, start, end, 'r', lw=2)
                if j + 1 < len(regions):
                    plt.hlines(i, end, regions[j+1][0], 'lightgray', lw=2)

            if regions[-1][1] < 1:
                plt.hlines(i, regions[-1][1], 1, 'lightgray', lw=2)

        score_data.extend([score for (score, _, _, _) in plot_data])

        ax = plt.gca()
        ax.set_yticks(ticks, labels)
        ax.set_position([0.05, 0.01, 0.9, 0.98])

        plt.title(f'{file}.parquet')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    # Show aggregate statistics across all parquet files
    plt.figure(figsize=(10, 10), dpi=80)
    counts, bins = np.histogram(score_data, bins=30)
    plt.stairs(counts, bins, fill=True)
    plt.title('Data quality score (DQS) distribution\n'
              'Including only examples where number of frames'
              ' > 2 * number of characters')
    plt.xlabel('DQS (formula: (3D/T + S/T) × 100, lower is better)\n'
               'T = total frames\n'
               f'D = dropped frames (more than {drop_threshold} missing '
               f'frames in a row)\n'
               'S = skipped frames (all other missing frames)')
    plt.ylabel(f'Number of examples (N = {len(score_data)})')
    plt.savefig(f'{output_dir}/score_dist.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    args = parse_args()
    check_args(args)
    metadata_file = (
        args.parquet_file.parents[1] /
        "metadata" /
        args.parquet_file.name
    ).with_suffix(".csv")

    dropped_frame_analysis(
        metadata_file,
        args.parquet_file,
        rh_only=True
    )
