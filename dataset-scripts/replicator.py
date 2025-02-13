import os
from tqdm import tqdm
import os

def build_lookup_table(directory):
    """
    Builds a lookup table (dictionary) mapping file names to their absolute paths in the given directory.
    """
    lookup = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.h5'):
                lookup[file] = os.path.join(root, file)
    return lookup

def generate_links_with_lookup(dir1, dir2, dir3, dir4):
    """
    Processes files in dir1 and creates hard links in dir4
    from dir2 or dir3 for .h5 files, preserving the directory structure.
    Uses lookup tables for efficient path resolution.
    """
    # Build lookup tables for dir2 and dir3
    dir2_lookup = build_lookup_table(dir2)
    dir3_lookup = build_lookup_table(dir3)

    for root, _, files in os.walk(dir1):
        rel_path = os.path.relpath(root, dir1)
        target_dir = os.path.join(dir4, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            if not file.endswith('.h5'):
                continue

            target_file = os.path.join(target_dir, file)

            # Find the source file path using lookup tables
            source_file = dir2_lookup.get(file) or dir3_lookup.get(file)

            if source_file:
                try:
                    if os.path.exists(target_file):
                        os.remove(target_file)
                    os.link(source_file, target_file)
                    yield (file, True, f"Linked {file} from {source_file} to {target_file}")
                except Exception as e:
                    yield (file, False, f"Error linking {file}: {e}")
            else:
                yield (file, False, f"File {file} not found in lookup tables for {dir2} or {dir3}")


def replicate_with_progress(dir1, dir2, dir3, dir4):
    """
    Uses tqdm to track the progress of creating hard links.
    """
    with open("replication.out", "w") as f:
        for i in tqdm(generate_links_with_lookup(dir1, dir2, dir3, dir4)):
            f.write(f"Copy {i[2]}\n")
            f.flush()


# Example usage
dir1 = '/data/sign_language_videos/popsign_v2/563_normalized_hands/'
dir2 = '/data/sign_language_videos/popsign_v2/313_hands_world_xyz_norm/'
dir3 = '/data/sign_language_videos/popsign_v2/250_hands_world_xyz_norm/'
dir4 = '/data/sign_language_videos/popsign_v2/563_hands_world_xyz_norm/'

replicate_with_progress(dir1, dir2, dir3, dir4)
