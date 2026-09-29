import argparse
import subprocess
from pathlib import Path
import pyarrow.parquet as pq


def get_video_frames(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "csv=p=0",
        str(video_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed:\n{result.stderr}"
        )

    return int(result.stdout.strip())


def main(dataset):

    dataset = Path(dataset)

    video_root = dataset / "videos"
    data_root = dataset / "data"


    print("=" * 60)
    print("Checking LeRobot v3.0 dataset")
    print(dataset)
    print("=" * 60)


    # ==========================
    # 1. Check videos
    # ==========================

    print("\n[Video frames]")

    video_total = {}

    for camera_dir in sorted(video_root.iterdir()):

        if not camera_dir.is_dir():
            continue

        total = 0

        for mp4 in camera_dir.rglob("*.mp4"):
            frames = get_video_frames(mp4)

            print(
                f"{camera_dir.name}/{mp4.name}: {frames}"
            )

            total += frames


        video_total[camera_dir.name] = total

        print(
            f"--> {camera_dir.name} total frames = {total}\n"
        )


    # ==========================
    # 2. Check parquet
    # ==========================

    print("=" * 60)
    print("[Parquet frames]")
    print("=" * 60)

    parquet_total = 0

    for f in sorted(data_root.rglob("*.parquet")):

        table = pq.read_table(f)

        print(
            f"{f.name}: {table.num_rows}"
        )

        parquet_total += table.num_rows


    print("\nTotal parquet frames =", parquet_total)


    # ==========================
    # Summary
    # ==========================

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for cam, frames in video_total.items():

        diff = frames - parquet_total

        print(
            f"{cam:45s} "
            f"video={frames:8d}, "
            f"parquet={parquet_total:8d}, "
            f"diff={diff:+d}"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        help="LeRobot v3 dataset path"
    )

    args = parser.parse_args()

    main(args.dataset)