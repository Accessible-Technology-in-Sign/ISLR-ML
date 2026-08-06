#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: args.py
Author: Rohit Sridhar
Date: 29-03-2026
Last Modified: Sun 29 Mar 2026 02:39:51 PM EDT
Version: X.X

Description:
    This file contains arguments shared among wireframe, overlay and frameview
    It also contains argument checking and custom is_required helpers.
"""
import argparse
import sys

from pathlib import Path
from utils import (
    WIREFRAME_SCRIPT,
    OVERLAY_SCRIPT,
    FRAMEVIEW_SCRIPT,
)

#################### Custom typing ####################
# rounded float in (0.0, 1.0]
def float_2p(f):
    try:
        f = round(float(f), 2)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Value {f} could not be converted to float of precision 2: {e}")

    if f <= 0.0 or f > 1.0:
        raise argparse.ArgumentTypeError(f"Value {f} must be in range (0.0, 1.0]")
    return f


#################### Customized requirements and helpers ####################
def is_calling_script(scripts):
    return Path(sys.argv[1]).name in scripts

def is_required(argname):
    """
    is_required does customized requirements per argument (flag) passed
    from the cmd line
    """
    if argname == "video_src":
        return is_calling_script([OVERLAY_SCRIPT])
    elif argname == "video_src":
        return is_calling_script([OVERLAY_SCRIPT, WIREFRAME_SCRIPT])
    elif argname == "parquet_file":
        return is_calling_script([OVERLAY_SCRIPT, WIREFRAME_SCRIPT, FRAMEVIEW_SCRIPT])

def check_args(args):
    if is_calling_script([OVERLAY_SCRIPT, WIREFRAME_SCRIPT]):
        if args.filter_pt_id is None and args.filter_video_dir is None:
            raise argparse.ArgumentTypeError("Pass exactly one of filter_pt_id or filter_video_dir")
        elif args.filter_pt_id is not None and args.filter_video_dir is not None:
            raise argparse.ArgumentTypeError("Do not pass both filter_pt_id and filter_video_dir")

#################### main arg parser ####################
def parse_args():
    """
    arg parser main function. import this to parse the args below. they are customized
    to scripts in the mputils/visualization/ dir
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-pt", "--filter_pt_id", type=str, required=is_required("filter_pt_id"), help="participant filtering. optional.")
    parser.add_argument("-fv", "--filter_video_dir", required=is_required("filter_video_dir"), type=Path, help="Filter sequences to only videos in dir.")
    parser.add_argument("-ss", "--sample_size", type=int, required=is_required("sample_size"), help="sample size.")
    parser.add_argument("-vs", "--video_src", type=Path, required=is_required("video_src"), help="directory with videos.")
    parser.add_argument("-pf", "--parquet_file", type=Path, required=is_required("parquet_file"), help="parquet file for processing.")
    parser.add_argument("-sd", "--seed", type=int, default=7987, help="seed for random sampling.")
    parser.add_argument("-bp", "--bar_position", type=int, default=0, help="Bar position for tqdm.")
    parser.add_argument("-ip", "--interpolate", type=int, default=None, help="turns on interpolation. only for wireframe videos.")
    parser.add_argument("-std", "--std_out", action="store_true", help="output to standard out (tqdm will be disabled).")
    parser.add_argument("-dbg", "--debug", action="store_true", help="output to standard out (tqdm will be disabled).")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    pass

