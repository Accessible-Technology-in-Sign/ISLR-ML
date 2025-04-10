import h5py as h5
import json
import os
import sys
import argparse
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
from pathlib import Path

ACTION_TYPES = [
    "print_signer_info",
    "make_parquet",
    "analyze_dropped_frames",
    "import_supplemental",
    "import_tfrecordio",
    "make_h5_table"
]

ARG_NAME_REQUIREMENTS = {
    "sort_by": {"print_signer_info"},
    "h5_loc": {"make_parquet", "make_h5_table"},
    "parquet_file": {"make_parquet", "analyze_dropped_frames", "import_supplemental"},
    "dropped_csv_file": {"analyze_dropped_frames"},
    "metadata_loc": {"make_parquet", "make_h5_table", "print_signer_info"},
    "data_loc": {"import_supplemental"}
}

global args

# dict_keys(['signingType', 'collectionId', 'clipFilename', 'phrase', 'signerId', 'fullVideoFilename', 'fullVideoFirstPtsTimeS', 'annotationStartTimeS', 'annotationEndTimeS', 'clipStartTimeS', 'clipEndTimeS', 'clipStartPacketMd5', 'clipEndPacketMd5', 'clipCreationTime', 'clipFileSize', 'clipFileMd5'])

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
        "--sort_by",
        choices=["time", "num_vids"],
        type=str,
        required=required_by_analysis_type("sort_by"),
        help="sort signer info output by param"
    )
    
    parser.add_argument(
        "--data_loc",
        type=Path,
        required=required_by_analysis_type("data_loc"),
        help="data location for importing kaggle data. should be a directory."
    )

    parser.add_argument(
        "--metadata_loc",
        type=Path,
        required=required_by_analysis_type("metadata_loc"),
        help="metadata location for h5 concatenation. pass this if using import_supplemental with metadata outside this repo."
    )

    parser.add_argument(
        "--h5_loc",
        type=Path,
        required=required_by_analysis_type("h5_loc"),
        help="h5 files location. will convert h5 files into one parquet file"
    )
    
    parser.add_argument(
        "--parquet_file",
        type=valid_parquet,
        required=required_by_analysis_type("parquet_file"),
        help="location for parquet. the meaning changes based on analysis_type"
    )
    
    parser.add_argument(
        "--dropped_csv_file",
        type=valid_csv,
        required=required_by_analysis_type("dropped_csv_file"),
        help="where to save dropped frames pct table. required from analyze_dropped_frames"
    )

    parser.add_argument(
        "--action_type",
        choices=ACTION_TYPES,
        type=str,
        required=True,
        help="action type. determines the analysis/data prep to run. warning: make_h5_table is unfinished."
    )

    return parser.parse_args()
    
################################################################################
############################ generic helpers ###################################
################################################################################
get_parquet_path = lambda : Path(args.parquet_file.parent)
get_dropped_csv_path = lambda : Path(args.dropped_csv_file.parent)
get_metadata_path = lambda pqp: Path(str(pqp).replace("landmarks", "metadata").replace(".parquet",".csv"))
get_dropped_hist_path = lambda dcsv: Path(str(dcsv).replace("analysis", "plots").replace(".csv",".png"))

def make_dirs():
    if args.action_type == "make_parquet":
        parquet_path = get_parquet_path()
        metadata_path = get_metadata_path(parquet_path)

        parquet_path.mkdir(parents=True, exist_ok=True)
        metadata_path.mkdir(parents=True, exist_ok=True)
    elif args.action_type == "analyze_dropped_frames":
        dropped_csv_path = get_dropped_csv_path()
        dropped_hist_path = get_dropped_hist_path(dropped_csv_path)

        dropped_csv_path.mkdir(parents=True, exist_ok=True)
        dropped_hist_path.mkdir(parents=True, exist_ok=True)
    elif args.action_type == "import_supplemental":
        parquet_path = get_parquet_path()
        parquet_path.mkdir(parents=True, exist_ok=True)
        
def get_column_names(hands):
    return [f"{coord}_{hand}_{idx}"
                for hand in hands
                for coord in ('x','y','z')
                for idx in list(range(21))]

def key_metadata(metadata):
    keyed_metadata = {}
    for datum in metadata:
        key = datum.pop("clipFilename")
        keyed_metadata[os.path.splitext(key)[0]] = datum
    return keyed_metadata

def get_metadata(metadata_file):
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    return metadata

################################################################################
#################### analyze_dropped_frames helpers ############################
################################################################################
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
        print(seq)
        print(f"NA Count: {na_count}")
        print(f"Top NA Count: {na_top_count}")
        print(f"Bottom NA Count: {na_bottom_count}")
        print()
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
def get_new_metadata(seq_id, h5_file, keyed_metadata, new_metadata):
    basename = os.path.splitext(h5_file)[0]

    pt_id = keyed_metadata[basename]['signerId']
    phrase = keyed_metadata[basename]['phrase']

    new_metadata.extend([[seq_id, pt_id, phrase]])

def get_new_metadata_filename():
    new_path_parts = list(args.parquet_file.parts)
    new_path_parts[-2] = "metadata"
    new_path_parts[-1] = new_path_parts[-1].replace(".parquet",".csv")
    new_path = Path(*new_path_parts)
    return new_path

def write_main_df(data):
    df = pd.DataFrame(data, columns=["sequence_id", "frame"] + get_column_names(("right",)))
    
    df = df.astype({"sequence_id":"int", "frame":"int"})
    df.iloc[:,2:] = df.iloc[:,2:].replace(0, np.nan)
    
    df = df.set_index("sequence_id")
    df.to_parquet(args.parquet_file)

def write_meta_df(metadata):
    meta_df = pd.DataFrame(metadata, columns=["sequence_id", "participant_id", "phrase"])

    meta_df = meta_df.astype({"sequence_id":"int"})
    meta_df = meta_df.set_index("sequence_id")
    
    new_metadata_file = get_new_metadata_filename()
    meta_df.to_csv(new_metadata_file)

################################################################################
############################## main methods ####################################
################################################################################
def print_signer_info():
    metadata = get_metadata(args.metadata_loc)
    signers = defaultdict(list)

    for clip in metadata:
        video_file_name = clip["fullVideoFilename"]
        vid_time = clip["annotationEndTimeS"] - clip["annotationStartTimeS"]
        signers[clip["signerId"]].append(vid_time)

    signer_info = []

    for signer,vid_times in signers.items():
        print(signer)
        signer_info.append((signer, sum(vid_times) / len(vid_times), len(vid_times)))

    i = 1 if args.sort_by == "time" else 2
    signer_info.sort(key=lambda x: x[i])

    print(f"Number of Signers: {len(signer_info)}")
    print(signer_info)

def make_parquet():
    h5_files = os.listdir(args.h5_loc)
    seq_id = 0
    
    data = []
    new_metadata = []
    
    metadata = get_metadata(args.metadata_loc)
    keyed_metadata = key_metadata(metadata)
    
    for h5_file in h5_files:
        out_file = os.path.join(args.h5_loc, h5_file)
        
        with h5.File(out_file, 'r') as h5_data:
            seq = np.array(h5_data["data"]).squeeze()
            seq = np.swapaxes(seq, 1, 2)
            seq = np.reshape(seq, (seq.shape[0], -1))
            
            seq_ids = np.tile([seq_id], seq.shape[0])
            seq = np.concatenate((seq_ids[:,None], seq), axis=1)
            
            seq = seq.tolist()
            for i,frame in enumerate(seq):
                frame.insert(1, i)
            
            get_new_metadata(seq_id, h5_file, keyed_metadata, new_metadata)
            data.extend(seq)
            seq_id += 1
            
    # Write main dataframe
    write_main_df(data)

    # Write meta dataframe
    write_meta_df(new_metadata)

def plot_hist_by_col(df):
    fig, axs = plt.subplots(3)

    axs[0].hist(df.loc[:,["na_middle_pct"]], bins=20)
    axs[0].set_title("Dropped Middle Frames (%)")
    axs[0].set(ylabel="Count")

    axs[1].hist(df.loc[:,["na_top_pct"]], bins=20)
    axs[1].set_title("Dropped Top Frames (%)")
    axs[1].set(ylabel="Count")

    axs[2].hist(df.loc[:,["na_bottom_pct"]], bins=20)
    axs[2].set_title("Dropped Bottom Frames (%)")
    axs[2].set(ylabel="Count")

    plt.tight_layout()

    dropped_hist_file = get_dropped_hist_path(args.dropped_csv_file)
    plt.savefig(dropped_hist_file)
    plt.close()

def analyze_dropped_frames():
    if not os.path.exists(args.parquet_file):
        raise FileNotFoundError("Error: create parquet file first.")
        return

    df = pd.read_parquet(args.parquet_file)
    if args.metadata_loc is None:
        metadata_path = get_metadata_path(args.parquet_file)
    else:
        metadata_path = args.metadata_loc
    metadata = pd.read_csv(metadata_path)

    participant_df = metadata.loc[:, ["sequence_id", "participant_id", "phrase"]].drop_duplicates()
    
    counts_df = get_counts_from_df(df)
    na_top_bottom_count = counts_df.loc[:,"na_top_count"] + counts_df.loc[:,"na_bottom_count"]
    counts_df["na_middle_count"] = counts_df.loc[:,"na_count"] - na_top_bottom_count
    
    counts_df = get_pct_of_total(counts_df, "na_top_count")
    counts_df = get_pct_of_total(counts_df, "na_bottom_count")
    counts_df = get_pct_of_total(counts_df, "na_middle_count")

    dropped_frames = pd.merge(counts_df, participant_df, on="sequence_id")
    dropped_frames = dropped_frames.loc[:,["participant_id", "na_top_pct", "na_bottom_pct", "na_middle_pct", "na_top_count", "na_bottom_count", "na_middle_count", "total_count"]]
    
    final = dropped_frames.groupby("participant_id").mean()
    final.to_csv(args.dropped_csv_file)

    plot_hist_by_col(final)

def import_supplemental_data():
    supp_char_map = os.path.join(args.data_loc , "supplemental_character_to_prediction_index.json")
    supp_landmarks = os.path.join(args.data_loc, "supplemental_landmarks")
    supp_metadata = os.path.join(args.data_loc, "supplemental_metadata.csv")

    # metadata = pd.read_csv(supp_metadata)
    landmarks_files = [os.path.join(supp_landmarks, supp_file) for supp_file in os.listdir(supp_landmarks)]

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
    df.to_parquet(args.parquet_file)
    
## TODO 
#   See if this can work. As of now, it looks
#   like all options involve looping over h5
##  sequences/files in python since there is 
#   metadata at the sequence level (e.g. the
#   phrase)
##
def make_h5_table():
    metadata = get_metadata(args.metadata_loc)
    metadata = key_metadata(metadata)
    
    h5_files = sorted(os.listdir(args.h5_loc))
    main_h5 = os.path.join(args.h5_loc, 'main.h5')

    random.seed(564)
    random.shuffle(h5_files)
    
    for h5_file in h5_files:
        f = h5.File(os.path.join('.', 'data', 'h5_data', h5_file), 'r')
        dset = f['intermediate']

        seq = dset[:].squeeze()
        seq = np.reshape(seq, (seq.shape[0], -1))
        
if __name__ == "__main__":
    args = parse_args()
    make_dirs()

    print(args)

    if args.action_type == "print_signer_info":
        print_signer_info()
    elif args.action_type == "make_h5_table":
        make_h5_table()
    elif args.action_type == "make_parquet":
        make_parquet()
    elif args.action_type == "analyze_dropped_frames":
        analyze_dropped_frames()
    elif args.action_type == "import_supplemental":
        import_supplemental_data()

