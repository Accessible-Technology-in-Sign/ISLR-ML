import cv2
import ffmpeg
from pathlib import Path

def is_video(filepath):
    return filepath.suffix == ".mp4"

def file_iters(filepath):
    if is_video(filepath):
        cap  = cv2.VideoCapture(filepath)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    else:
        length = len(list(filepath.glob("*.jpg")))
    return length

def load_video(filepath):
    # metadata = ffmpeg.probe(filepath)['streams'][0]['tags']
    # if 'rotate' in metadata:
    #     rotate = True
    # else:
    #     rotate = False
    
    cap  = cv2.VideoCapture(filepath)
    print(f"Cap Orientation: {int(cap.get(cv2.CAP_PROP_ORIENTATION_META))}")
    rotate = int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) == 270
    if not cap.isOpened():
        raise IOError(f"Could not open Video File at {filepath}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        pre_rot_frame_shape = frame.shape
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        post_rot_frame_shape = frame.shape
        
        # cv2.imwrite(frame_name, frame)
        yield frame, pre_rot_frame_shape, post_rot_frame_shape
    cap.release()

def load_jpgs(filepath):
    jpgs = list(filepath.glob("*.jpg"))
    jpgs.sort()

    for jpg in jpgs:
        img = cv2.imread(jpg)

        if img is None:
            raise IOError(f"Error: Image {jpg} not found or unable to read.")

        yield img


def load_file(filepath):
    if is_video(filepath):
        yield from load_video(filepath)
    else:
        yield from load_jpgs(filepath)

