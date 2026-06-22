from collections import Counter
from pathlib import Path

import numpy as np
import os
import argparse
import importlib

def main():
    parser = argparse.ArgumentParser(description="Process video with a specified mediapipe solution.")

    parser.add_argument('solution', type=str, help="The solution to run.")
    parser.add_argument('input_file', type=Path, help="Path to the input file.")
    parser.add_argument('output_file', type=Path, help="Path to the output file.")

    parser.add_argument('--loader', type=str, default='cv2_loader', help="The name of the loader module (default: cv2_loader).")
    parser.add_argument('--writer', type=str, default='h5py_writer', help="The name of the writer module (default: h5py_writer).")

    args = parser.parse_args()

    loader_module = importlib.import_module(args.loader)
    load_file = getattr(loader_module, 'load_file')
    file_iters = getattr(loader_module, 'file_iters')

    solution_module = importlib.import_module(args.solution)
    run = getattr(solution_module, 'run')
    post_process = getattr(solution_module, 'post_process')

    writer_module = importlib.import_module(args.writer)
    write_file = getattr(writer_module, 'write_file')

    results = np.zeros((file_iters(args.input_file), 63)) #TODO, make rest dynamic
    print(f"Landmarks Shape: {results.shape}")

    ctr = Counter({"lh":0,"rh":0})
    pre_rot_shapes = set()
    post_rot_shapes = set()

    for idx, (frame, pre_rot_shape, post_rot_shape) in enumerate(load_file(args.input_file)):
        # results[idx, :, :, :] = post_process(run(frame), world=False)
        pre_rot_shapes.add(pre_rot_shape)
        post_rot_shapes.add(post_rot_shape)
        frame_name = '_'.join(args.input_file.stem.split('_')[:2] + [str(idx)]) + '.png'
        frame_name = args.output_file.parent.joinpath(frame_name)

        # post_proc_dict = {'flipH':True, 'frame_name': frame_name}
        post_proc_dict = {'flipH':True}
        results[idx, :], left_handed = post_process(run(frame, **post_proc_dict), world=False)
        if left_handed is not None:
            ctr["lh"] += left_handed
            ctr["rh"] += not(left_handed)
    print(f"Input File: {args.input_file}")
    print(f"Pre Rot Frame Shape (check orientation): {pre_rot_shapes}")
    print(f"Post Rot Frame Shape (check orientation): {post_rot_shapes}")
    left_handed = ctr["lh"] > ctr["rh"]
    write_file(args.output_file, results, left_handed=left_handed)

if __name__ == "__main__":
    import sys
    __file__ = "mputil"
    sys.argv[0] = "mputil"
    main()

