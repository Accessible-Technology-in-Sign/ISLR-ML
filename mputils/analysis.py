#!/home/rsridhar37/miniconda3/envs/mp_extraction/bin/python

import ffmpeg
import shutil
import cv2
import logging
import json
import os
import sys
import argparse
import random
import re
import h5py as h5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import csv
import glob
import hashlib
import struct
import typing

import pyarrow
import pyarrow.parquet

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

BINS = [x/100 for x in range(0,10,1)] + [x/100 for x in range(10,100,10)]

ACTION_TYPES = [
    "analyze_word_info",
    "print_signer_info",
    "make_parquet",
    "analyze_dropped_frames",
    "import_supplemental",
    "import_tfrecordio",
    "make_h5_table",
    "segment_data",
    "sample_videos_by_orientation",
]

_FEATURE_SIZES = {
    'timestamp': 1,
    'pose': (33, 25),
    'left_hand': 21,
    'right_hand': 21,
    'face': 478,
}

ARG_NAME_REQUIREMENTS = {
    "sort_by": {"print_signer_info"},
    "h5_loc": {"make_parquet", "make_h5_table"},
    "save_parquet_file": {"make_parquet", "import_supplemental"},
    "parquet_loc": {"import_tfrecordio"},
    "parquet_file": {"segment_data", "analyze_dropped_frames", "analyze_word_info"},
    "metadata_file": {"make_parquet", "make_h5_table", "print_signer_info"},
    "data_loc": {"import_supplemental"},
    "tfrecordio_loc": {"import_tfrecordio"},
    "participants": {"segment_data"},
    "video_loc": {"sample_videos_by_orientation"},
    "save_video_loc": {"sample_videos_by_orientation"},
}

ANALYSIS_COLUMNS = ["na_top_pct", "na_bottom_pct", "na_middle_pct", "na_top_count", "na_bottom_count", "na_middle_count", "total_count"]

# Used for segment_data function
WORD_ID = 0
CORRECT_WORDS = 0
INCORRECT_WORDS = 0
OUTPUT_MLF_PARENT=Path(f"/data/hmm_modeling/fingerspelling/ContinuousBigram/ext/supplemental_gen_drop-na_lininterp0/dim20/thr0/test/pt")

FILE_SUFFIXES= {
    "metadata":".csv",
    "landmarks":".parquet",
    "analysis":".csv",
    "plots":".png",
}

global args

################################################################################
############################ argparse helpers ##################################
################################################################################
def valid_csv(arg_value):
    pattern=r".*\/analysis\/.+\.csv"
    csv_file_pattern = re.compile(pattern)
    if csv_file_pattern.fullmatch(arg_value) is None:
        raise ValueError("csv file must have parent dir analysis and end with a .csv ext")
    return Path(arg_value)

def valid_parquet(arg_value):
    pattern=r".*\/landmarks\/.+\.parquet"
    parquet_file_pattern = re.compile(pattern)
    if parquet_file_pattern.fullmatch(arg_value) is None:
        raise ValueError("Parquet file must have parent dir landmarks and end with a .parquet ext")
    return Path(arg_value)

def required_by_analysis_type(argname):
    if "--help" in sys.argv:
        return False

    i = sys.argv.index("--action_type") + 1
    return sys.argv[i] in ARG_NAME_REQUIREMENTS[argname]

################################################################################
################################ argparse ######################################
################################################################################
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--action_type",
        choices=ACTION_TYPES,
        type=str,
        required=True,
        help="action type. determines the analysis/data prep to run. warning: make_h5_table is unfinished."
    )
    
    parser.add_argument(
        "--video_loc",
        type=Path,
        required=required_by_analysis_type("video_loc"),
        help="video location for sampling original videos."
    )

    parser.add_argument(
        "--save_video_loc",
        type=Path,
        required=required_by_analysis_type("save_video_loc"),
        help="location to save sampled videos. will copy from video loc to this location"
    )

    parser.add_argument(
        "--data_loc",
        type=Path,
        required=required_by_analysis_type("data_loc"),
        help="data location for importing kaggle data. should be a directory."
    )

    parser.add_argument(
        "--h5_loc",
        type=Path,
        required=required_by_analysis_type("h5_loc"),
        help="h5 files location. will convert h5 files into one parquet file"
    )

    parser.add_argument(
        "--tfrecordio_loc",
        type=Path,
        required=required_by_analysis_type("tfrecordio_loc"),
        help="location for parquet. the meaning changes based on analysis_type"
    )
    
    parser.add_argument(
        "--parquet_loc",
        type=Path,
        required=required_by_analysis_type("parquet_loc"),
        help="location for parquet. use it for data split among multiple parquet files."
    )

    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Optional log file for logging. If None, logs to stdout. All parent directories will be creatied if non existent."
    )

    parser.add_argument(
        "--metadata_file",
        type=Path,
        required=required_by_analysis_type("metadata_file"),
        help="metadata file for h5 concatenation. pass this if using metadata outside this repo. if doing seq_level analyis, no need to pass it."
    )

    parser.add_argument(
        "--parquet_file",
        type=Path,
        required=required_by_analysis_type("parquet_file"),
        help="parquet file path. used for reading in parquet files created by this script."
    )

    parser.add_argument(
        "--save_parquet_file",
        type=valid_parquet,
        required=required_by_analysis_type("save_parquet_file"),
        help="location for parquet. the meaning changes based on analysis_type"
    )

    parser.add_argument(
        "--participants",
        type=str,
        nargs="+",
        required=required_by_analysis_type("participants"),
        help="Participants to make segmentations for",
    )

    parser.add_argument(
        "--sort_by",
        choices=["time", "fpl"],
        type=str,
        required=required_by_analysis_type("sort_by"),
        help="sort signer info output by param"
    )
    
    parser.add_argument(
        "--sample_method",
        type=str,
        default=None,
        choices=[None, "random", "nan_range"],
        help="sample method for make_parquet"
    )
    
    parser.add_argument(
        "--rng",
        type=float,
        nargs=2,
        default=None,
        help="NaN range for nan_range sample method"
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="sample size when sampling using sample method/sample video by orientation"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=564,
        help="sample size when sampling using sample method/sample video by orientation"
    )
    
    parser.add_argument(
        "--verify_frame_count",
        action="store_true",
        help="Generate videos with subtitles. Otherwise generates the segmented dataset. Only for segment_data.",
    )

    parser.add_argument(
        "--seq_level",
        action="store_true",
        help="pass this to perform dropped frame analysis on seq level"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="pass this to perform dropped frame analysis on seq level"
    )

    return parser.parse_args()
    
################################################################################
############################ generic helpers ###################################
################################################################################

# set up the logger
def setup_logger():
    if args.log_file is None:
        logging.basicConfig(
            stream=sys.stdout,
            level=logging.DEBUG if args.debug else logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d | %H:%M:%S"
        )
    else:
        args.log_file.parent.mkdir(exist_ok=True, parents=True)
        logging.basicConfig(
            filename=args.log_file,
            filemode='w',
            level=logging.DEBUG if args.debug else logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d | %H:%M:%S"
        )

def get_file_name(old_filename, ft_to, dir_only=False):
    if ft_to not in set(FILE_SUFFIXES.keys()):
        raise ValueError(f"get_file_name is being called with incorrect ft_to value: {ft_to}")

    new_parts = list(old_filename.parts)
    new_parts[1] = ft_to
    
    if dir_only:
        new_filename = Path(*new_parts)
    else:
        new_filename = Path(*new_parts).with_suffix(FILE_SUFFIXES[ft_to])

    return new_filename

# get_metadata_file = lambda pqp: Path(str(pqp).replace("landmarks", "metadata").replace(".parquet",".csv"))
# get_dropped_hist_file = lambda dcsv: Path(str(dcsv).replace("analysis", "plots").replace(".csv",".png"))

def get_dropped_csv_file(parquet_path):
    ext = "_dropped_seql.csv" if args.seq_level else "_dropped.csv"
    return Path(str(parquet_path).replace("landmarks", "analysis").replace(".parquet", ext))

def make_dirs():
    if args.action_type == "make_parquet":
        parquet_path = Path(args.save_parquet_file.parent)
        # this works because is passed in rather than a file
        # despite the weird semantics
        # metadata_path = get_metadata_file(parquet_path)
        metadata_path = get_file_name(parquet_path, "metadata", dir_only=True)

        parquet_path.mkdir(parents=True, exist_ok=True)
        metadata_path.mkdir(parents=True, exist_ok=True)
    elif args.action_type == "analyze_dropped_frames":
        dropped_csv_path = Path(get_dropped_csv_file(args.parquet_file).parent)
        # dropped_hist_path = get_dropped_hist_file(dropped_csv_path)
        dropped_hist_path = get_file_name(dropped_csv_path, "plots", dir_only=True)

        dropped_csv_path.mkdir(parents=True, exist_ok=True)
        dropped_hist_path.mkdir(parents=True, exist_ok=True)
    elif args.action_type == "import_supplemental":
        parquet_path = Path(args.save_parquet_file.parent)
        parquet_path.mkdir(parents=True, exist_ok=True)
        
def get_column_names(hands):
    return [f"{coord}_{hand}_{idx}"
                for hand in hands
                for coord in ('x','y','z')
                for idx in list(range(21))]

def key_metadata(metadata):
    keyed_metadata = {}
    for datum in metadata:
        key = Path(datum.pop("clipFilename"))
        keyed_metadata[key.stem] = datum
    return keyed_metadata

def get_metadata(metadata_file):
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    return metadata

################################################################################
########################## segment_data helpers ################################
################################################################################

# Gets the best recommendation from the MLF file (the first
# recommendation in the file) for each sequence
def get_best_rec(recs, pt_metadata, seq_recs, seq_start_frames):
    # Invidual recs delimited by "///\n"
    rec_list = recs.split("///\n")
    best_rec = rec_list[0]

    rec_lines = best_rec.split("\n")[:-1]
    rec_path = Path(rec_lines[0][1:-1])
    
    seq_id = int(rec_path.stem)
    if seq_id not in seq_start_frames:
        logging.warning(seq_id)
        return
    
    seq_recs[seq_id] = []
    for segment in rec_lines[1:]:
        segment_split = segment.split(" ")
        
        start_ms = int(int(segment_split[0]) / 1000)
        if start_ms > 0:
            start_ms += seq_start_frames[seq_id]["start_frame"]
        rec_word = segment_split[2]
        
        seq_recs[seq_id].append((rec_word, start_ms))

# One iteration of gather_mlf_results.
# Parses mlf file, gets the top rec
# and gathers the frame segmentation
# for the video
def gather_mlf_result(mlf, pt_metadata, seq_recs, seq_start_frames):
    with open(mlf, "r") as f:
        lines = f.readlines()

    # First join lines. In MLFs, rec lists
    # delimited by ".\n"
    lines = "".join(lines[1:-1])
    recs_by_seq = lines.split(".\n")
    
    for recs in recs_by_seq:
        get_best_rec(recs, pt_metadata, seq_recs, seq_start_frames)

# Look in output_mlf_path to gather MLF word results
def gather_mlf_results(pt_metadata, participant, seq_recs, seq_start_frames):
    output_mlf_path = OUTPUT_MLF_PARENT.joinpath(participant)
    word_mlfs = list(output_mlf_path.glob("result.mlf_word.*"))
    for mlf in word_mlfs:
        result = gather_mlf_result(mlf, pt_metadata, seq_recs, seq_start_frames)

# Gets the start frame for each sequence, since
# there may be nan values excluded in the MLF 
# frame counts
def get_seq_start_frames(pt_metadata, pt_landmarks, seq_start_frames):
    def get_start_frame(row):
        seq_id = row.index.to_list()[0]
        first_col_vals = row.x_right_0.fillna(0).to_list()
        if sum(first_col_vals) == 0:
            return

        i = 0
        while first_col_vals[i] == 0:
            i += 1

        seq_start_frames[seq_id] = {
            "start_frame": i,
            "num_frames": len(first_col_vals)
        }
        return

    pt_landmarks.groupby("sequence_id").apply(get_start_frame)

# Gather segmented data from original sequence phrase data. Creates
# landmarks and metadata on the word level
def gather_data(seq_id, seq_recs, seq_landmarks, seq_metadata):
    corr_phrase = seq_metadata.phrase.split(" ")
    pred_and_frames = seq_recs[seq_id][1:]
    # pred_and_frames = seq_recs[seq_id]
    
    gathered_metadata = []
    gathered_landmarks = []
    
    global WORD_ID, CORRECT_WORDS, INCORRECT_WORDS
    for i in range(len(pred_and_frames) - 1):
        if i < len(corr_phrase) and corr_phrase[i] == pred_and_frames[i][0]:
        # if True:
            word_frame_start = pred_and_frames[i][1]
            word_frame_end = pred_and_frames[i+1][1]

            gathered_metadata.append([
                WORD_ID,
                seq_id,
                seq_metadata.participant_id,
                # pred_and_frames[i][0],
                corr_phrase[i],
                seq_metadata.clipFilename,
                word_frame_start,
                word_frame_end
            ])

            word_landmarks = seq_landmarks[(seq_landmarks.frame >= word_frame_start) & (seq_landmarks.frame < word_frame_end)]
            logging.debug(f"word start: {word_frame_start}")
            logging.debug(f"word end: {word_frame_end}")
            logging.debug(f"word: {pred_and_frames[i][0]}")
            logging.debug(f"{word_landmarks}")
            word_landmarks = word_landmarks.reset_index().values.tolist()

            for i in range(len(word_landmarks)):
                word_landmarks[i][0] = WORD_ID
                word_landmarks[i][1] = i
            gathered_landmarks.extend(word_landmarks)

            WORD_ID += 1
            CORRECT_WORDS += 1
        else:
            INCORRECT_WORDS += 1
    
    logging.debug(f"word start: {pred_and_frames[-1][1]}")
    logging.debug(f"word: {pred_and_frames[-1][0]}")
    logging.debug(f"{seq_landmarks[seq_landmarks.frame >= pred_and_frames[-1][1]]}")
    return gathered_metadata, gathered_landmarks

# Segments each participant's data by segmenting their landmarks.
def segment_pt(
    participant,
    seq_recs,
    seq_start_frames,
    pt_metadata,
    pt_landmarks,
    word_metadata,
    word_landmarks,
):
    get_seq_start_frames(pt_metadata, pt_landmarks, seq_start_frames)
    mlf_results = gather_mlf_results(pt_metadata, participant, seq_recs, seq_start_frames)
    
    segmented_seqs = list(seq_recs.keys())
    for seq_id in tqdm(segmented_seqs):
        seq_landmarks = pt_landmarks.loc[seq_id]
        seq_metadata = pt_metadata.loc[seq_id]

        gathered_metadata, gathered_landmarks = gather_data(
            seq_id,
            seq_recs,
            seq_landmarks,
            seq_metadata,
        )

        word_metadata.extend(gathered_metadata)
        word_landmarks.extend(gathered_landmarks)
        
################################################################################
####################### analyze_word_info helpers ##############################
################################################################################
def save_word_analysis_histograms(df, plt_prefix=""):
    # Some dataframe housekeeping
    df["fpv"] = df["num_frames"] / df["num_vids"]
    df["lpv"] = df["num_letters"] / df["num_vids"]
    df = df.drop(["num_frames", "num_letters"], axis=1)
    
    # Get the file name from parts
    file_parts = [args.parquet_file.stem, plt_prefix, "hist"]
    parent = get_file_name(args.parquet_file.parent, "plots", dir_only=True)
    plt_file = parent.joinpath("_".join(file_parts)).with_suffix(FILE_SUFFIXES["plots"])

    # Save first batch of histograms
    _ = df.hist()
    plt.savefig(plt_file)
    plt.close("all")

    # Do lower 0.25 quartile and upper 0.75 quartile
    lq_df = df[df.num_vids <= df.num_vids.quantile(0.25)]
    uq_df = df[df.num_vids > df.num_vids.quantile(0.25)]

    file_parts_lq = file_parts + ["l25"]
    file_parts_uq = file_parts + ["u75"]
    plt_file_lq = parent.joinpath("_".join(file_parts_lq)).with_suffix(FILE_SUFFIXES["plots"])
    plt_file_uq = parent.joinpath("_".join(file_parts_uq)).with_suffix(FILE_SUFFIXES["plots"])

    _ = lq_df.hist()
    plt.savefig(plt_file_lq)
    plt.close("all")

    _ = uq_df.hist()
    plt.savefig(plt_file_uq)
    plt.close("all")

def save_word_analysis_csvs(df, analysis_prefix=""):
    fig, ax = plt.subplots(2,2)

    fpl_df = df.sort_values("fpl", ascending=False).head(600)[["fpl"]]
    num_vids_df = df.sort_values("num_vids", ascending=False).head(600)[["num_vids"]]

    for i,metric in enumerate(["fpl", "num_vids"]):
        file_parts = [args.parquet_file.stem, analysis_prefix, metric]
        parent = get_file_name(args.parquet_file.parent, "analysis", dir_only=True)
        analysis_file = parent.joinpath("_".join(file_parts)).with_suffix(FILE_SUFFIXES["analysis"])

        analysis_df_var = "_".join([metric,"df"])
        analysis_df = locals()[analysis_df_var]
        analysis_df.to_csv(analysis_file)

def save_word_analysis(df, prefix=""):
    df["fpl"] = df.num_frames / df.num_letters
    save_word_analysis_csvs(df, analysis_prefix=prefix)
    save_word_analysis_histograms(df, plt_prefix=prefix)

################################################################################
#################### analyze_dropped_frames helpers ############################
################################################################################
def plot_hist_by_col(df, dropped_csv_file):
    fig, axs = plt.subplots(3)

    axs[0].hist(df.loc[:,["na_middle_pct"]], bins=BINS)
    axs[0].set_title("Dropped Middle Frames (%)")
    axs[0].set(ylabel="Count")

    axs[1].hist(df.loc[:,["na_top_pct"]], bins=BINS)
    axs[1].set_title("Dropped Top Frames (%)")
    axs[1].set(ylabel="Count")

    axs[2].hist(df.loc[:,["na_bottom_pct"]], bins=BINS)
    axs[2].set_title("Dropped Bottom Frames (%)")
    axs[2].set(ylabel="Count")

    plt.tight_layout()

    # dropped_hist_file = get_dropped_hist_file(dropped_csv_file)
    dropped_hist_file = get_file_name(dropped_csv_file, "plots")
    plt.savefig(dropped_hist_file)
    plt.close("all")

def get_counts(seq):
    total_count = seq.shape[0]
    na_count = seq.x_right_0.isna().sum()
    if na_count == total_count:
        return total_count, total_count, total_count, 0

    na_vals = seq.x_right_0.isna().tolist()

    if na_vals[0]:
        na_top_count = na_vals.index(0)
    else:
        na_top_count = 0

    na_vals.reverse()
    if na_vals[0]:
        na_bottom_count = na_vals.index(0)
    else:
        na_bottom_count = 0
    
    if na_top_count + na_bottom_count > na_count:
        logging.warning(seq)
        logging.warning(f"NA Count: {na_count}")
        logging.warning(f"Top NA Count: {na_top_count}")
        logging.warning(f"Bottom NA Count: {na_bottom_count}")
        logging.warning()
    return total_count, na_count, na_top_count, na_bottom_count

def get_counts_from_df(df):
    counts = df.groupby("sequence_id").apply(get_counts)
    counts = counts.to_frame(name="all")

    counts[["total_count", "na_count", "na_top_count", "na_bottom_count"]] = counts.iloc[:,0].to_list()
    counts = counts.drop(labels="all", axis=1)
    counts.index.name = "sequence_id"

    return counts

def get_pct_of_total(counts_df, col):
    new_col = col.replace("count","pct")
    counts_df[new_col] = counts_df[col] / counts_df["total_count"]
    return counts_df

################################################################################
######################## make_parquet helpers ##################################
################################################################################
# Sorts h5 files by na values
def sort_by_na_vals(h5_files):
    if args.rng is None:
        raise ValueError("Need to pass rng arg when using nan_range method")

    new_h5_files = []
    for h5_file in tqdm(h5_files, desc="Sort by NA Vals"):
        with h5.File(h5_file, 'r') as h5_data:
            seq = np.array(h5_data["data"])
            if seq.shape[0] < 10:
                continue

            nan_rows = seq.sum(axis=1) == 0
            nan_pct = nan_rows.sum() / seq.shape[0]
            if args.rng[0] <= nan_pct <= args.rng[1]:
                new_h5_files.append((h5_file, nan_pct))
    
    new_h5_files.sort(key=lambda x: x[1], reverse=True)
    new_h5_files = [x[0] for x in new_h5_files]
    return new_h5_files

# Get a filtered, sorted list of h5 files 
# ready for processing by mputils
def get_filtered_list(h5_files):
    if args.participants is not None:
        h5_files = [h5_file for h5_file in h5_files if h5_file.stem.startswith(args.participants[0])]
    
    h5_files.sort()
    # Skip the case where sample_method is random and just return it later.
    if args.sample_method is None:
        random.shuffle(h5_files)
        return h5_files
    elif args.sample_method == "nan_range":
        h5_files = sort_by_na_vals(h5_files)

    participants = set([h5_file.stem.split('_')[0] for h5_file in h5_files])
    logging.info(f"Number of Videos: {len(h5_files)}")
    logging.info(f"Participants: {participants}")

    random.shuffle(h5_files)
    if len(h5_files) <= args.sample_size:
        return h5_files
    else:
        return random.sample(h5_files, args.sample_size)

# Creates new metadata for the new parquet data
def get_new_metadata(seq_id, h5_file, keyed_metadata, new_metadata):
    basename = h5_file.stem

    pt_id = keyed_metadata[basename]['signerId']
    phrase = keyed_metadata[basename]['phrase']
    filename = basename + ".mp4"

    new_metadata.extend([[seq_id, pt_id, phrase, filename]])

# Writes the main parquet data frame
def write_main_df(data, parquet_file):
    df = pd.DataFrame(data, columns=["sequence_id", "frame"] + get_column_names(("right",)))
    
    df = df.astype({"sequence_id":"int", "frame":"int"})
    df.iloc[:,2:] = df.iloc[:,2:].replace(0, np.nan)
    
    df = df.set_index("sequence_id")
    df.to_parquet(parquet_file, engine="fastparquet")

# Writes metadata csv file
def write_meta_df(
    metadata,
    parquet_file,
    columns=["sequence_id", "participant_id", "phrase", "clipFilename"],
    coltypes={"sequence_id":"int"}
):
    meta_df = pd.DataFrame(metadata, columns=columns)

    meta_df = meta_df.astype(coltypes)
    meta_df = meta_df.set_index("sequence_id")
    
    # new_metadata_file = get_metadata_file(parquet_file)
    new_metadata_file = get_file_name(parquet_file, "metadata")
    meta_df.to_csv(new_metadata_file)

################################################################################
###################### import_tfrecordio helpers ###############################
################################################################################
def get_schema():
    """Get a pyarrow Schema for the parquet tables of landmarks."""
    schema = pyarrow.schema([])
    schema = schema.append(pyarrow.field('clip_id', pyarrow.int32()))
    schema = schema.append(pyarrow.field('frame', pyarrow.int32()))
    for t in ['x', 'y', 'z', 'presence', 'visibility']:
        for position in ['pose', 'left_hand', 'right_hand', 'face']:
          size = _FEATURE_SIZES[position]
          if isinstance(size, tuple):
              size = size[0]
          for i in range(size):
              schema = schema.append(
                pyarrow.field(f'{t}_{position}_{i}', pyarrow.float32()))
    schema = schema.append(
        pyarrow.field('timestamp', pyarrow.int64()))
    return schema

def sequence_example_to_table(example, schema, video_level_data):
    """Convert a sequence_example into a pyarrow Table."""
    output = list()
    num_frames = example.context.feature['num_frames'].int64_list.value[0]
    video_name = example.context.feature['video_name'].bytes_list.value[0]
    # Create a positive int32 from the video_name as the identifier for the
    # example.  TODO(mgeorg) we should ensure this is unique.
    digest = hashlib.md5(video_name, usedforsecurity=False)
    ident_bytes = digest.digest()[:4]
    ident_bytes = ident_bytes[:3] + (
        ident_bytes[3] & 0x7f).to_bytes(1, byteorder='little')
    ident = struct.unpack('<i', ident_bytes)[0]
    video_level_data[ident] = dict()
    for k, v in example.context.feature.items():
        if v.WhichOneof('kind') == 'int64_list':
            video_level_data[ident][k] = v.int64_list.value[0]
        elif v.WhichOneof('kind') == 'bytes_list':
            video_level_data[ident][k] = v.bytes_list.value[0].decode('utf-8')
        elif v.WhichOneof('kind') == 'float_list':
            video_level_data[ident][k] = v.float_list.value[0]
    for frame_number in range(num_frames):
        output.append({'clip_id': ident, 'frame': frame_number})
    for k, feature_list in example.feature_lists.feature_list.items():
        assert len(feature_list.feature) == num_frames
        for frame_number in range(num_frames):
            feature = feature_list.feature[frame_number]
            m = re.match(
                r'^(x|y|z|presence|visibility)_(pose|left_hand|right_hand|face)$', k)
            if m:
                target_len = _FEATURE_SIZES[m.group(2)]
            else:
                assert k == 'timestamp'
                target_len = 1
            kind = feature.WhichOneof('kind')
            if kind is None:
                # Feature is empty.
                continue
            actual_len = len(getattr(feature, kind).value)
            if isinstance(target_len, tuple):
                assert actual_len in target_len, {
                    'k': k, 'target_len': target_len, 'actual_len': actual_len}
            else:
                assert actual_len == target_len, {
                    'k': k, 'target_len': target_len, 'actual_len': actual_len}
            for i in range(actual_len):
                if k == 'timestamp':
                    assert i == 0
                    assert kind == 'int64_list', feature
                    output[frame_number][k] = (
                        feature.int64_list.value[i])
                    continue
                assert kind == 'float_list', feature
                output[frame_number][f'{k}_{i}'] = (
                    feature.float_list.value[i])
    return pyarrow.Table.from_pylist(output, schema=schema)

class VideoLevelData(typing.NamedTuple):
    """A row in the video level data csv."""
    clip_id: int | None
    video_name: str | None
    prompt: str | None
    num_frames: int | None
    image_width: int | None
    image_height: int | None
    image_frame_rate: float | None

################################################################################
############################## main methods ####################################
################################################################################

# Prints word info, from the word dataset
def analyze_word_info():
    metadata_file = get_file_name(args.parquet_file, "metadata")
    metadata = pd.read_csv(metadata_file)

    metadata["num_frames"] = metadata.frame_end - metadata.frame_start
    metadata["num_vids"] = 1
    metadata["num_letters"] = metadata.phrase.str.len()
    
    meta_analysis = metadata[["participant_id", "phrase", "num_frames", "num_vids", "num_letters"]]
    meta_analysis_pt = meta_analysis.drop("phrase", axis=1).groupby(["participant_id"]).sum()
    meta_analysis_phrase = meta_analysis.drop("participant_id", axis=1).groupby(["phrase"]).sum()
    
    save_word_analysis(meta_analysis_pt, prefix="by_pt")
    save_word_analysis(meta_analysis_phrase, prefix="by_phrase")

# Prints signer info, such as the number of signers
# and their fpl/video times
def print_signer_info():
    metadata = get_metadata(args.metadata_file)
    signers = defaultdict(list)

    for clip in metadata:
        video_file_name = clip["fullVideoFilename"]
        vid_time = clip["annotationEndTimeS"] - clip["annotationStartTimeS"]
        signers[clip["signerId"]].append(vid_time)

    signer_info = []

    for signer,vid_times in signers.items():
        signer_info.append((signer, sum(vid_times) / len(vid_times), len(vid_times)))

    i = 1 if args.sort_by == "fpl" else 2
    signer_info.sort(key=lambda x: x[i])

    logging.info(f"Number of Signers: {len(signer_info)}")
    for info in signer_info:
        logging.info(info)

# Make parquet files from H5 data
def make_parquet():
    seq_id = 0
    
    data = []
    new_metadata = []
    
    metadata = get_metadata(args.metadata_file)
    keyed_metadata = key_metadata(metadata)
    
    h5_files = list(args.h5_loc.iterdir())
    h5_files = get_filtered_list(h5_files)

    for h5_file in tqdm(h5_files, desc="Make Parquet File"):
        with h5.File(h5_file, 'r') as h5_data:
            seq = np.array(h5_data["data"])
            
            seq_ids = np.tile([seq_id], seq.shape[0])
            seq = np.concatenate((seq_ids[:,None], seq), axis=1)
            
            seq = seq.tolist()
            for i,frame in enumerate(seq):
                frame.insert(1, i)
            
            get_new_metadata(seq_id, h5_file, keyed_metadata, new_metadata)
            data.extend(seq)
            seq_id += 1
            
    # Write main dataframe
    write_main_df(data, args.save_parquet_file)

    # Write meta dataframe
    write_meta_df(new_metadata, args.save_parquet_file)

# Runs dropped frames analysis. Outputs histogram
# of pct dropped frames (either by participant or
# sequence, depending on the args)
def analyze_dropped_frames():
    if not args.parquet_file.exists():
        raise FileNotFoundError("Error: create parquet file first.")
        return

    df = pd.read_parquet(args.parquet_file)

    counts_df = get_counts_from_df(df)
    na_top_bottom_count = counts_df.loc[:,"na_top_count"] + counts_df.loc[:,"na_bottom_count"]
    counts_df["na_middle_count"] = counts_df.loc[:,"na_count"] - na_top_bottom_count
    
    counts_df = get_pct_of_total(counts_df, "na_top_count")
    counts_df = get_pct_of_total(counts_df, "na_bottom_count")
    counts_df = get_pct_of_total(counts_df, "na_middle_count")

    if args.seq_level:
        final = counts_df.reset_index().loc[:,["sequence_id"] + ANALYSIS_COLUMNS]
        final = final.set_index("sequence_id")
    else:
        if args.metadata_file is None:
            # metadata_file = get_metadata_file(args.parquet_file)
            metadata_file = get_file_name(args.parquet_file, "metadata")
        else:
            metadata_file = args.metadata_file

        metadata = pd.read_csv(metadata_file)
        participant_df = metadata.loc[:, ["sequence_id", "participant_id", "phrase"]].drop_duplicates()

        dropped_frames = pd.merge(counts_df, participant_df, on="sequence_id")
        dropped_frames = dropped_frames.loc[:,["participant_id"] + ANALYSIS_COLUMNS]

        final = dropped_frames.groupby("participant_id").mean()

    dropped_csv_file = get_dropped_csv_file(args.parquet_file)
    final.to_csv(dropped_csv_file)
    plot_hist_by_col(final, dropped_csv_file)

# Import supplemental data (in the parquet files) from the kaggle competition
def import_supplemental():
    supp_char_map = args.data_loc.joinpath("supplemental_character_to_prediction_index.json")
    supp_landmarks = args.data_loc.joinpath("supplemental_landmarks")
    supp_metadata = args.data_loc.joinpath("supplemental_metadata.csv")

    # metadata = pd.read_csv(supp_metadata)
    landmarks_files = [supp_landmarks.joinpath(supp_file) for supp_file in supp_landmarks.iterdir()]

    cols = ["sequence_id", "frame"] + get_column_names(("right_hand",))
    df = pd.DataFrame(columns=cols)
    for lm_file in landmarks_files:
        df_next = pd.read_parquet(lm_file)
        df_next = df_next.reset_index()
        
        df_subset = df_next.loc[:,cols]
        df = pd.concat([df, df_subset])
    
    # metadata = metadata.loc[:,["sequence_id","participant_id"]].drop_duplicates().reset_index(drop=True)
    df.columns = [col.replace("_hand","") for col in cols]

    df = df[["sequence_id", "frame"] + get_column_names(("right",))].set_index("sequence_id")
    df.to_parquet(args.save_parquet_file)
    
# Import tfrecordio files from the kaggle competition
def import_tfrecordio():
    import tensorflow as tf

    schema = get_schema()
    filenames = sorted(glob.glob(str(args.tfrecordio_loc.joinpath("*.tfrecordio-?????-of-?????"))))
    # filenames = sorted(glob.glob('*/landmarks/*.tfrecordio-?????-of-?????'))
    all_video_level_data = dict()
    args.parquet_loc.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        path = Path(filename)
        m = re.match(r'^([^-]*)-([^-]*).tfrecordio(-\d{5}-of-\d{5})$', path.name)
        if not m:
            logging.warning(f'Unable to parse filename {filename}')
            continue
        study = m.group(1)
        split = m.group(2)
        shard_str = m.group(3)
        all_video_level_data[f'{study}-{split}'] = all_video_level_data.get(
            f'{study}-{split}', dict())
        video_level_data = all_video_level_data[f'{study}-{split}']
        parquet_path = args.parquet_loc.joinpath(f'{study}-{split}.parquet{shard_str}')
        # parquet_path = path.parent.joinpath(f'{study}-{split}.parquet{shard_str}')
        raw_dataset = tf.data.TFRecordDataset(filename)
        tables = list()
        for raw in raw_dataset:
            example = tf.train.SequenceExample()
            example.ParseFromString(raw.numpy())
            video_name = example.context.feature[
                'video_name'].bytes_list.value[0].decode('utf-8')
            logging.info(video_name)
            tables.append(sequence_example_to_table(
                example, schema, video_level_data))
        if not tables:
            logging.warning(f'No entries for table {parquet_path}')
            continue
        logging.info(f'Writing to table {parquet_path}')
        table = pyarrow.concat_tables(tables)
        pyarrow.parquet.write_table(table, parquet_path)
    # Create the supplemental csv files.
    for study_split in all_video_level_data:
        study, split = study_split.split('-')
        supplemental_csv = f'{study}/landmarks/{study}-{split}_supplemental.csv'
        logging.info(f'Writing supplemental video level information to {supplemental_csv}')
        with open(supplemental_csv, 'w', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(VideoLevelData._fields)
            rows = list()
            for clip_id, video_data in all_video_level_data[study_split].items():
                rows.append(VideoLevelData(
                    clip_id=clip_id,
                    video_name=video_data.get('video_name'),
                    prompt=video_data.get('prompt'),
                    num_frames=video_data.get('num_frames'),
                    image_width=video_data.get('image/width'),
                    image_height=video_data.get('image/height'),
                    image_frame_rate=video_data.get('image/frame_rate'),
                ))
            rows.sort(key=lambda x: (x.video_name or '', x.prompt or ''))
            writer.writerows(rows)

## TODO (INCOMPLETE)
#   See if this can work. As of now, it looks
#   like all options involve looping over h5
##  sequences/files in python since there is 
#   metadata at the sequence level (e.g. the
#   phrase)
##
def make_h5_table():
    metadata = get_metadata(args.metadata_file)
    metadata = key_metadata(metadata)
    
    h5_files = sorted(list(args.h5_loc.iterdir()))
    main_h5 = args.h5_loc.joinpath('main.h5')

    random.shuffle(h5_files)
    
    for h5_file in h5_files:
        f = h5.File(Path('.').joinpath('data', 'h5_data', h5_file), 'r')
        dset = f['intermediate']

        seq = dset[:].squeeze()
        seq = np.reshape(seq, (seq.shape[0], -1))

# Segments the data and produces a new dataset of isolated, fingerspelled
# words
def segment_data():
    word_metadata = []
    word_landmarks = []

    # metadata_file = get_metadata_file(args.parquet_file)
    metadata_file = get_file_name(args.parquet_file, "metadata")
    metadata = pd.read_csv(metadata_file, index_col="sequence_id")
    landmarks = pd.read_parquet(args.parquet_file)

    for participant in args.participants:
        # seq recs contains the segments with the NaN
        # padding for the beginning included
        seq_recs = {}
        # seq start frames contains frame counts and 
        # start frame (NaN padding)
        seq_start_frames = {}
        
        pt_metadata = metadata[metadata.participant_id == participant]
        pt_seq_ids = pt_metadata.index.to_list()
        pt_landmarks = landmarks.loc[pt_seq_ids]

        segment_pt(
            participant,
            seq_recs,
            seq_start_frames,
            pt_metadata,
            pt_landmarks,
            word_metadata,
            word_landmarks,
        )
    
    new_metadata_cols = ["sequence_id"] + metadata.reset_index().columns.to_list() + ["frame_start", "frame_end"]
    new_metadata_cols[1] = "orig_sequence_id"
    new_metadata_coltypes = {"sequence_id":"int", "orig_sequence_id":"int"}

    new_parquet_file = Path(args.parquet_file.parent).joinpath(args.parquet_file.stem + "_word").with_suffix(".parquet")
    write_meta_df(
        word_metadata,
        new_parquet_file,
        columns=new_metadata_cols,
        coltypes=new_metadata_coltypes
    )
    write_main_df(word_landmarks, new_parquet_file)

    logging.debug(f"Correct Words: {CORRECT_WORDS}")
    logging.debug(f"Incorrect Words: {INCORRECT_WORDS}")

def sample_videos_from_orientation_dict(orientation_dict):
    for orientation in orientation_dict:
        orientation_count = len(orientation_dict[orientation])
        logging.info(f"[Orientation|Count]: {orientation}|{orientation_count}")

        if args.sample_size > 0:
            sampled_videos = orientation_dict[orientation][:args.sample_size]
        else:
            sampled_videos = orientation_dict[orientation]
        # logging.info(f"Sampled Videos: {sampled_videos}")

        dest = args.save_video_loc / f"orientation-{str(orientation)}"
        dest.mkdir(exist_ok=True, parents=True)
        for video in sampled_videos:
            dest_f = dest / video.name
            logging.debug(f"{dest_f=}")
            os.link(video, dest_f)
            # shutil.copy2(video, dest)

def sample_videos_by_orientation():
    video_files = list(args.video_loc.glob("*.mp4"))
    random.shuffle(video_files)

    videos_by_cap_orientation = defaultdict(list)
    videos_by_meta_orientation = defaultdict(list)
    videos_by_frame_shape = defaultdict(list)

    for video_file in tqdm(video_files):
        cap = cv2.VideoCapture(video_file)
        cap_orientation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))

        metadata = ffmpeg.probe(video_file)['streams'][0]['tags']
        meta_orientation = "rotate" in metadata

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        videos_by_cap_orientation[f"cap{cap_orientation}"].append(video_file)
        videos_by_meta_orientation[f"meta{meta_orientation}"].append(video_file)
        videos_by_frame_shape[f"fshp{frame_width}-{frame_height}"].append(video_file)

    sample_videos_from_orientation_dict(videos_by_cap_orientation)
    sample_videos_from_orientation_dict(videos_by_meta_orientation)
    sample_videos_from_orientation_dict(videos_by_frame_shape)

if __name__ == "__main__":
    args = parse_args()

    make_dirs()
    random.seed(args.seed)
    
    setup_logger()
    logging.info(args)
    
    current_module = sys.modules[__name__]
    analysis_func = getattr(current_module, args.action_type)
    analysis_func()

    # if args.action_type == "analyze_word_info":
    #     analyze_word_info()
    # elif args.action_type == "print_signer_info":
    #     print_signer_info()
    # elif args.action_type == "make_h5_table":
    #     make_h5_table()
    # elif args.action_type == "make_parquet":
    #     make_parquet()
    # elif args.action_type == "analyze_dropped_frames":
    #     analyze_dropped_frames()
    # elif args.action_type == "import_supplemental":
    #     import_supplemental()
    # elif args.action_type == "import_tfrecordio":
    #     import_tfrecordio()
    # elif args.action_type == "segment_data":
    #     segment_data()
    # elif args.action_type == ""

