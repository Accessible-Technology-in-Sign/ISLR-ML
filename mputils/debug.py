import h5py as h5
import numpy as np

from pathlib import Path

#################### This script functions as a debugging playground ####################

def load_file(h5_filepath):
    if not(isinstance(h5_filepath, Path)):
        raise TypeError("pass Path to load_file")

    with h5.File(h5_filepath, 'r') as h5_data:
        seq = np.array(h5_data["data"])

    return seq

def print_seq(seq, start, end):
    for i in range(seq.shape[0]):
        print(seq[i, start:end])

if __name__ == "__main__":
    

