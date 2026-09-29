import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from carm_py import carm_py as carm
import time
from typing import List

TARGET_KEYS = ["action", "observation.state"]


def quat2rot6d_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    """
    quat_xyzw: shape (4,), [qx, qy, qz, qw]
    return: shape (6,), [r00, r10, r20, r01, r11, r21]
    """
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)

    R = Rotation.from_quat(quat_xyzw).as_matrix()

    rot6d = np.concatenate(
        [
            R[:, 0],  # 第一列: r00, r10, r20
            R[:, 1],  # 第二列: r01, r11, r21
        ],
        axis=0,
    )

    return rot6d.astype(np.float32)


def pose7_to_pose9(pose7: np.ndarray) -> np.ndarray:
    """
    pose7: [x, y, z, qx, qy, qz, qw]
    return: [x, y, z, r00, r10, r20, r01, r11, r21]
    """
    pose7 = np.asarray(pose7, dtype=np.float32)

    xyz = pose7[:3]
    quat_xyzw = pose7[3:7]

    rot6d = quat2rot6d_xyzw(quat_xyzw)

    return np.concatenate([xyz, rot6d], axis=0).astype(np.float32)


def left_arm_fk(left_q6: np.ndarray) -> np.ndarray:
    """
    输入:
        left_q6: shape (6,), 左臂 6 个关节角

    输出:
        left_pose7: shape (7,), [x, y, z, qx, qy, qz, qw]

    这里必须替换成你自己的左臂正运动学。
    """
    return left_arm.forward_kine(0, left_q6)[-1]

def right_arm_fk(right_q6: np.ndarray) -> np.ndarray:
    """
    输入:
        right_q6: shape (6,), 右臂 6 个关节角

    输出:
        right_pose7: shape (7,), [x, y, z, qx, qy, qz, qw]

    这里必须替换成你自己的右臂正运动学。
    """
    return right_arm.forward_kine(0, right_q6)[-1]


# def forward_kine(tool_index : int, jnt_value : List[float]):
#     """
#     单点正解
#     tool_index: 工具号
#     jnt_value: 关节值vector
#     返回(ret, quat_pose)
#     """
#     ret, pose = carm_.forward_kine(tool_index, jnt_value)
#     print(f"forward_kine, ret = {ret}")
#     print("quat_pose =", pose)
#     return ret, pose

def convert_joint14_to_ee20(x: np.ndarray) -> np.ndarray:
    """
    输入:
        x: shape (14,)
           [left_j1...left_j6, left_gripper,
            right_j1...right_j6, right_gripper]

    输出:
        y: shape (20,)
           [left_x, left_y, left_z, left_rot6d(6), left_gripper,
            right_x, right_y, right_z, right_rot6d(6), right_gripper]
    """
    x = np.asarray(x, dtype=np.float32)

    if x.shape[-1] != 14:
        raise ValueError(f"Expected input shape (14,), but got {x.shape}")

    left_q6 = x[0:6]
    left_gripper = x[6:7]

    right_q6 = x[7:13]
    right_gripper = x[13:14]

    left_pose7 = left_arm_fk(left_q6)
    right_pose7 = right_arm_fk(right_q6)

    left_pose9 = pose7_to_pose9(left_pose7)
    right_pose9 = pose7_to_pose9(right_pose7)

    y = np.concatenate(
        [
            left_pose9,
            left_gripper,
            right_pose9,
            right_gripper,
        ],
        axis=0,
    )

    return y.astype(np.float32)


def convert_cell(v):
    """
    parquet 单元格里通常是 list 或 np.ndarray。
    这里默认每个单元格是 14 维。
    """
    arr = np.asarray(v, dtype=np.float32)

    if arr.shape == (14,):
        return convert_joint14_to_ee20(arr).tolist()

    raise ValueError(f"Expected one cell with shape (14,), but got {arr.shape}")


def process_one_parquet(src_path: Path, dst_path: Path):
    df = pd.read_parquet(src_path)

    for key in TARGET_KEYS:
        if key not in df.columns:
            print(f"[WARN] {src_path} 没有字段 {key}，跳过")
            continue

        print(f"  converting key: {key}")
        df[key] = df[key].apply(convert_cell)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_path, index=False)

    print(f"[OK] {src_path} -> {dst_path}")


def process_dir(input_dir: str, output_dir: str, overwrite: bool = False):
    input_dir = Path(input_dir)

    parquet_files = sorted(input_dir.rglob("*.parquet"))

    if len(parquet_files) == 0:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    print(f"Found {len(parquet_files)} parquet files.")

    if overwrite:
        for src_path in parquet_files:
            process_one_parquet(src_path, src_path)
    else:
        output_dir = Path(output_dir)

        for src_path in parquet_files:
            rel_path = src_path.relative_to(input_dir)
            dst_path = output_dir / rel_path
            process_one_parquet(src_path, dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", default="/home/xxy/.cache/huggingface/lerobot/eval/folding_280_processed/data/chunk-000", type=str)
    parser.add_argument("--output_dir", type=str, default="/home/xxy/Documents/DATA/")
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    right_arm = carm.CArmSingleCol("10.42.0.101")
    left_arm = carm.CArmSingleCol("10.42.0.103")
    time.sleep(1.0)

    if not args.overwrite and args.output_dir is None:
        raise ValueError("不覆盖原文件时，必须指定 --output_dir")

    process_dir(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )

