import json
import h5py
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

METADATA = "/data/sign_language_videos/fingerspelling_videos/dmk_v1/metadata/dmk_v1-train.json"
VIDEO_DIR = "/data/sign_language_videos/fingerspelling_videos/dmk_v1/video_clips"
# METADATA = "/storage/home/hcoda1/5/rsridhar37/scratch/fingerspelling_videos/dmk_v1/metadata/dmk_v1-train.json"
# VIDEO_DIR = "/storage/home/hcoda1/5/rsridhar37/scratch/fingerspelling_videos/dmk_v1/video_clips"

ARG_NAME_REQUIREMENTS = {
    "sort_by": {"print_signer_info"},
    "h5_loc": {"make_parquet"},
    "parquet_file": {"make_parquet", "analyze_dropped_frames"}, # Add back in later, "import_supplemental"},
    "dropped_csv_file": {"analyze_dropped_frames"}
}

SUPP_CHAR_MAP = "/data/parquet/asl-fingerspelling/supplemental_character_to_prediction_index.json"
SUPP_LANDMARKS = "/data/parquet/asl-fingerspelling/supplemental_landmarks"
SUPP_METADATA = "/data/parquet/asl-fingerspelling/supplemental_metadata.csv"

global args

# dict_keys(['signingType', 'collectionId', 'clipFilename', 'phrase', 'signerId', 'fullVideoFilename', 'fullVideoFirstPtsTimeS', 'annotationStartTimeS', 'annotationEndTimeS', 'clipStartTimeS', 'clipEndTimeS', 'clipStartPacketMd5', 'clipEndPacketMd5', 'clipCreationTime', 'clipFileSize', 'clipFileMd5'])

def required_by_analysis_type(argname):
    if "--help" in sys.argv:
        return False

    i = sys.argv.index("--action_type") + 1
    return sys.argv[i] in ARG_NAME_REQUIREMENTS[argname]

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
        "--h5_loc",
        type=str,
        required=required_by_analysis_type("h5_loc"),
        help="h5 files location. will convert h5 files into one parquet file"
    )
    
    parser.add_argument(
        "--parquet_file",
        type=str,
        required=required_by_analysis_type("parquet_file"),
        help="location for parquet. the meaning changes based on analysis_type"
    )
    
    parser.add_argument(
        "--dropped_csv_file",
        type=str,
        required=required_by_analysis_type("dropped_csv_file"),
        help="where to save dropped frames pct table. required from analyze_dropped_frames"
    )

    parser.add_argument(
        "--action_type",
        choices=["print_signer_info", "make_parquet", "analyze_dropped_frames", "import_supplemental"],
        type=str,
        required=True,
        help="action type. determines the analysis/data prep to run"
    )

    return parser.parse_args()

def print_signer_info():
    with open(METADATA, "r") as f:
        metadata = json.load(f)

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

def get_column_names(hands):
    return [f"{coord}_{hand}_{idx}"
                for idx in list(range(21))
                for coord in ('x','y','z')
                for hand in hands]

def make_parquet():
    h5_files = os.listdir(args.h5_loc)
    seq_id = 0
    data = []

    for h5_file in h5_files:
        out_file = os.path.join(args.h5_loc, h5_file)
        pt_id = h5_file.split('_')[0]

        with h5py.File(out_file, 'r') as h5_data:
            seq = np.array(h5_data["intermediate"]).squeeze()
            seq = np.reshape(seq, (seq.shape[0], -1))
            
            seq_ids = np.tile([seq_id], seq.shape[0])
            seq = np.concatenate((seq_ids[:,None], seq), axis=1)
            
            seq = seq.tolist()
            for i,frame in enumerate(seq):
                frame.insert(1, pt_id)
                frame.insert(2, i)

            data.extend(seq)
            seq_id += 1

    df = pd.DataFrame(data, columns=["sequence_id", "participant_id", "frame"] + get_column_names(("right",)))
    
    df = df.astype({"sequence_id":"int", "frame":"int"})
    df.iloc[:,3:] = df.iloc[:,3:].replace(0, np.nan)

    df = df.set_index("sequence_id")
    print(df)
    df.to_parquet(args.parquet_file)

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

def analyze_dropped_frames():
    if not os.path.exists(args.parquet_file):
        raise FileNotFoundError("Error: create parquet file first.")
        return

    df = pd.read_parquet(args.parquet_file)

    participant_df = df.reset_index()
    participant_df = participant_df.loc[:, ["sequence_id", "participant_id"]].drop_duplicates()
    # participant_df = participant_df.loc[:, ["sequence_id", "participant_id"]].drop_duplicates().reset_index(drop=True)
    # df = df.set_index("sequence_id")
    
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
    
    plt.hist(final.loc[:,["na_middle_pct"]], bins=20)
    plt.savefig(os.path.splitext(args.dropped_csv_file)[0] + ".png")

def import_supplemental_data():
    metadata = pd.read_csv(SUPP_METADATA)
    landmarks_files = [os.path.join(SUPP_LANDMARKS, supp_file) for supp_file in os.listdir(SUPP_LANDMARKS)]

    cols = ["sequence_id", "frame"] + get_column_names(("right_hand",))
    df = pd.DataFrame(columns=cols)
    for lm_file in landmarks_files:
        df_next = pd.read_parquet(lm_file)
        df_next = df_next.reset_index()
        
        df_subset = df_next.loc[:,cols]
        df = pd.concat([df, df_subset])
    
    metadata = metadata.loc[:,["sequence_id","participant_id"]].drop_duplicates().reset_index(drop=True)
    df.columns = [col.replace("_hand","") for col in cols]
    df = df.set_index("sequence_id")

    df = pd.merge(df, metadata, on="sequence_id")
    df = df[["sequence_id", "participant_id", "frame"] + get_column_names(("right",))].set_index("sequence_id")
    df.to_parquet("./out/supplemental.parquet")

if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.action_type == "print_signer_info":
        print_signer_info()
    elif args.action_type == "make_parquet":
        make_parquet()
    elif args.action_type == "analyze_dropped_frames":
        analyze_dropped_frames()
    elif args.action_type == "import_supplemental":
        import_supplemental_data()

