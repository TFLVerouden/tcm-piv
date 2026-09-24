#!/usr/bin/env python3
"""
Split a folder of .tiff files into multiple sibling folders, each capped at
MAX_TIFFS_PER_FOLDER tiffs.

- Tiffs are MOVED out of the source folder and distributed (alphabetically)
  across the new folders.
- Non-tiff files are COPIED ("duplicated") into every new folder, and also
  left in the original source folder.
- New folders sit next to the source folder and are named after it with a
  letter suffix: "P-001" -> "P-001a", "P-001b", ... "P-001z", "P-001aa", ...

Example:
    P-001/                      P-001a/
      img0001.tif                 img0001.tif
      img0002.tif                 img0002.tif
      readme.txt         -->      readme.txt   (copied)
      ...                      P-001b/
                                   img00xx.tif
                                   ...
                                   readme.txt   (copied)
"""

from pathlib import Path
import shutil
import string
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from tcm_utils.file_dialogs import ask_directory
from tcm_utils.io_utils import prompt_yes_no

# ----------------------------------------------------------------------
# CONFIG - edit these two lines
MAX_TIFFS_PER_FOLDER = 4096
COPY_WORKERS = 16
# ----------------------------------------------------------------------

TIFF_EXTENSIONS = {".tif", ".tiff"}


def letter_suffix(index: int) -> str:
    """0 -> 'a', 1 -> 'b', ..., 25 -> 'z'. Only single letters a-z are supported."""
    if index >= len(string.ascii_lowercase):
        raise SystemExit(
            "More than 26 folders would be needed (ran out of single letters a-z). "
            "Increase MAX_TIFFS_PER_FOLDER to reduce the number of folders."
        )
    return string.ascii_lowercase[index]


def main():

    source = ask_directory(
        key="split_tiff_folder_source",
        title="Select source folder containing .tiff files to split",
    )
    if source is None:
        raise SystemExit("No source folder selected.")

    all_files = sorted(p for p in source.iterdir() if p.is_file())
    tiff_files = [p for p in all_files if p.suffix.lower() in TIFF_EXTENSIONS]
    other_files = [p for p in all_files if p.suffix.lower()
                   not in TIFF_EXTENSIONS]

    if not tiff_files:
        raise SystemExit("No .tiff/.tif files found in the source folder.")

    # ceiling division
    num_folders = -(-len(tiff_files) // MAX_TIFFS_PER_FOLDER)
    chunks = [
        tiff_files[i * MAX_TIFFS_PER_FOLDER: (i + 1) * MAX_TIFFS_PER_FOLDER]
        for i in range(num_folders)
    ]

    dest_names = [
        f"{source.name}{letter_suffix(i)}" for i in range(num_folders)]

    print(
        f"Found {len(tiff_files)} tiff(s) and {len(other_files)} other file(s).")
    print(f"This will create {num_folders} folder(s) next to '{source.name}':")
    for name, chunk in zip(dest_names, chunks):
        print(f"  {name}  ({len(chunk)} tiff(s))")
    if not prompt_yes_no("Proceed? (press ENTER to confirm)"):
        print("Aborted - no files were moved or copied.")
        return

    dest_folders = []
    for name in dest_names:
        dest = source.parent / name
        dest.mkdir(exist_ok=True)
        dest_folders.append(dest)

    total_ops = len(tiff_files) + len(other_files) * num_folders
    with tqdm(total=total_ops, desc="Splitting files", unit="file") as bar:
        # Move each tiff into its assigned folder
        for dest, chunk in zip(dest_folders, chunks):
            for tiff_path in chunk:
                shutil.move(str(tiff_path), str(dest / tiff_path.name))
                bar.update(1)

        # Copy every non-tiff file into every new folder, in parallel
        # (original keeps its copy too)
        copy_jobs = [
            (other_path, dest / other_path.name)
            for dest in dest_folders
            for other_path in other_files
        ]
        with ThreadPoolExecutor(max_workers=COPY_WORKERS) as executor:
            futures = [
                executor.submit(shutil.copy2, str(src), str(dst))
                for src, dst in copy_jobs
            ]
            for future in as_completed(futures):
                future.result()  # re-raise if any copy failed
                bar.update(1)

    print(f"\nDone. Created {num_folders} folder(s):")
    for dest, chunk in zip(dest_folders, chunks):
        print(
            f"  {dest.name}: {len(chunk)} tiff(s) + {len(other_files)} other file(s)")


if __name__ == "__main__":
    main()
