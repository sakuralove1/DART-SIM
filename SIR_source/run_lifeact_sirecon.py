import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PROJECT_ROOT / "Demo"
DEFAULT_DATASET_PATH = Path(r"D:\Project\Python\SIM\SSRSIM\test\LifeAct DataSet")
TARGET_RAW_NAME = "seq1_TIRF-SIM488_GreenCh-DL.mrc"


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SIR_core.common import post_process_file  # noqa: E402


def iter_lifeact_raw_files(dataset_path):
    dataset_path = Path(dataset_path)
    for dirpath, _, fnames in sorted(os.walk(dataset_path)):
        for fname in sorted(fnames):
            if fname == TARGET_RAW_NAME:
                yield Path(dirpath) / fname


def sirecon_noisy_raw_data_for_training(path=DEFAULT_DATASET_PATH, ifprint=True, dry_run=False, limit=None):
    """Run the packaged SIR reconstruction step used by Demo_ZeroShot_LifeAct.py."""
    raw_files = list(iter_lifeact_raw_files(path))
    if limit is not None:
        raw_files = raw_files[:limit]

    print(f"dataset: {Path(path)}")
    print(f"matched raw files: {len(raw_files)}")

    for index, raw_file in enumerate(raw_files, start=1):
        print(f"[{index}/{len(raw_files)}] {raw_file}")
        if dry_run:
            continue
        previous_cwd = Path.cwd()
        try:
            os.chdir(DEMO_DIR)
            post_process_file(file=str(raw_file), imaging_device="MultiSIM002", ifprint=ifprint)
        finally:
            os.chdir(previous_cwd)

    return raw_files


def parse_args():
    parser = argparse.ArgumentParser(description="Run packaged SIR preprocessing for LifeAct data.")
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default=str(DEFAULT_DATASET_PATH),
        help="LifeAct dataset folder.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only list matched raw files.")
    parser.add_argument("--quiet", action="store_true", help="Disable verbose SIR txt/stdout printing.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N matched files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sirecon_noisy_raw_data_for_training(
        path=args.dataset_path,
        ifprint=not args.quiet,
        dry_run=args.dry_run,
        limit=args.limit,
    )
