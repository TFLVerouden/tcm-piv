from __future__ import annotations

from pathlib import Path

from tcm_piv.run_batch import run_batch


CONFIG_FILE = Path("source/tcm_piv/config/260831_config.toml")

FOLDERS = [
    Path("/Volumes/Data/PIV/260820_piv/step_0-5bar/260909_095041_step_0-5bar_14-5mA"),
    Path("/Volumes/Data/PIV/260820_piv/step_0-5bar/260909_095906_step_0-5bar_15-5mA"),
    Path("/Volumes/Data/PIV/260820_piv/step_1-0bar/260909_091412_step_1-0bar_15-5mA"),
    Path("/Volumes/Data/PIV/260820_piv/step_1-0bar/260909_091941_step_1-0bar_14-5mA"),
    Path("/Volumes/Data/PIV/260820_piv/step_1-0bar/260909_092539_step_1-0bar_13-5mA"),
    Path("/Volumes/Data/PIV/260820_piv/step_2-5bar/260909_112035_step_2-5bar_13-5mA"),
    Path("/Volumes/Data/PIV/260820_piv/step_2-5bar/260909_112635_step_2-5bar_14-5mA"),
    Path("/Volumes/Data/PIV/260820_piv/step_2-5bar/260909_113255_step_2-5bar_15-5mA"),
    Path("/Volumes/Data/PIV/260820_piv/step_3-0bar/260909_113930_step_3-0bar_14-5mA"),
    Path("/Volumes/Data/PIV/260820_piv/step_3-0bar/260909_115136_step_3-0bar_15-5mA"),
]


def main() -> None:
    successful_folders: list[Path] = []
    failed_folders: list[tuple[Path, Exception]] = []

    for folder in FOLDERS:
        try:
            print(f"Running batch for {folder}")
            run_batch(top_dir=folder, config_file=CONFIG_FILE)
            successful_folders.append(folder)
        except Exception as exc:
            print(f"Failed for {folder}: {exc}")
            failed_folders.append((folder, exc))
            continue

    print("\nBatch summary")
    print("Successful folders:")
    for folder in successful_folders:
        print(f"- {folder}")

    print("Failed folders:")
    for folder, exc in failed_folders:
        print(f"- {folder}: {exc}")


if __name__ == "__main__":
    main()
