import argparse
import glob
import os
import re


def get_dir_size(path):
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def extract_val_loss(filename):
    match = re.search(r"val_loss_(.*?)\.pt", filename)
    if match:
        return float(match.group(1))
    return float("inf")


def cleanup_checkpoints(checkpoint_dir):
    initial_size = get_dir_size(checkpoint_dir)
    pt_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))

    no_loss_file = None
    val_loss_files = []

    for file in pt_files:
        if "no_loss" in file:
            no_loss_file = file
        elif "val_loss" in file:
            val_loss_files.append(file)

    val_loss_files.sort(key=extract_val_loss)

    files_to_keep = val_loss_files[:5]
    if no_loss_file:
        files_to_keep.append(no_loss_file)

    for file in pt_files:
        if file not in files_to_keep:
            print(f"Deleting: {file}")
            os.remove(file)
        else:
            print(f"Keeping: {file}")

    final_size = get_dir_size(checkpoint_dir)
    saved_size = initial_size - final_size
    return saved_size


def format_size(size_bytes):
    """Convert size in bytes to human readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean up model checkpoints keeping only the 5 best validation loss files and no_loss file."
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Path to the directory containing model checkpoint files",
    )
    args = parser.parse_args()

    total_saved = 0
    if os.path.isdir(args.checkpoint_dir):
        for root, dirs, files in os.walk(args.checkpoint_dir):
            if any(f.endswith(".pt") for f in files):
                print(f"\nProcessing directory: {root}")
                saved = cleanup_checkpoints(root)
                total_saved += saved
                print(f"Saved {format_size(saved)} in {root}")
    else:
        saved = cleanup_checkpoints(args.checkpoint_dir)
        total_saved = saved
        print(f"Saved {format_size(saved)} in {args.checkpoint_dir}")

    print(f"\nTotal storage saved: {format_size(total_saved)}")
