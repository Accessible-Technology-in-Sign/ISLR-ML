#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filename: utils.py
Author: Rohit Sridhar
Date: 2026-03-27
Last Modified: Sun 29 Mar 2026 02:30:29 PM EDT
Version: 1.0

Description:
    This file contains shared utility functions and constants used
    in at least one of frameview.py, wireframe.py, and overlay.py
"""

import logging

from pathlib import Path

########## Useful constants
WIREFRAME_SCRIPT=Path("./wireframe.py").resolve()
OVERLAY_SCRIPT=Path("./wireframe.py").resolve()
FRAMEVIEW_SCRIPT=Path("./frameview.py").resolve()

WIREFRAME_PATH = Path("videos/wireframe").resolve()
OVERLAY_PATH = Path("videos/overlay").resolve()

FPS=30

########## Misc utils
def get_filter_stem(
    filter_video_dir=None,
    filter_pt_id=None,
):
    """
    get stem using filter args passed from cmd line
    """
    if filter_video_dir is not None:
        return filter_video_dir.parts[-1]
    elif filter_pt_id is not None:
        return filter_pt_id

def get_tqdm_filter_desc(
    dataset,
    filter_video_dir=None,
    filter_pt_id=None,
):
    """
    uses the two filtration criteria and generates a 
    description for tqdm progress bar.
    """
    stem = get_filter_stem(
        filter_video_dir=filter_video_dir,
        filter_pt_id=filter_pt_id
    )
    return f"{dataset}-{stem}"

########## Video path utils
def get_video_path(
    parquet_file,
    root,
    filter_video_dir=None,
    filter_pt_id=None,
):
    """
    gets the video path given the filter video dir and
    filter_pt_id.
    """
    stem_dir = get_filter_stem(
        filter_video_dir=filter_video_dir,
        filter_pt_id=filter_pt_id
    )
    video_dirname = f"{parquet_file.stem}"

    video_path = root / video_dirname / stem_dir
    video_path.mkdir(parents=True, exist_ok=True)

    return Path(video_path)

########## Logger set up utils
def setup_logger(
        seed,
        sample_size,
        parquet_file,
        logs_path,
        filter_pt_id=None,
        filter_video_dir=None,
        std_out=False,
        debug=False,
    ):
    """
    Sets up logger with name based on seed, sample size and
    whether filtering by video dir/pt id. Finally, outputs 
    to either log file or stdout in write mode.
    """
    log_filename = f"sd{seed}_n{sample_size}"
    if filter_pt_id is not None:
        log_filename += f"_pt-{filter_pt_id}"
    if filter_video_dir is not None:
        filter_video_dirname = filter_video_dir.parts[-1]
        log_filename += f"_{filter_video_dirname}"
    
    if std_out:
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d | %H:%M:%S'
        )
    else:
        if logs_path is None:
            raise ValueError("if std_out is False, must pass a logs_path")
        log_file = (
            logs_path /
            parquet_file.stem /
            log_filename
        ).with_suffix(".log")

        log_file.parent.mkdir(parents=True, exist_ok=True)
        filemode='w'
        logging.basicConfig(
            filename=log_file,
            filemode=filemode,
            level=logging.DEBUG if debug else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d | %H:%M:%S'
        )

if __name__ == "__main__":
    pass

