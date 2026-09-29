import numpy as np
import glob
import pyarrow.parquet as pq

ROOT = "/root/data1/xxy/huggingface/lerobot/g1/pen_100"

q01 = 5.399991214752197
q99 = 5.399991214752197

files = sorted(
    glob.glob(ROOT + "/**/*.parquet", recursive=True)
)

actions = []

for f in files:
    table = pq.read_table(f, columns=["action"])
    actions.append(
        np.asarray(
            table["action"].to_pylist(),
            dtype=np.float64
        )
    )

actions = np.concatenate(actions, axis=0)

x = actions[:, 15]

x_norm = (
    (x - q01)
    / (q99 - q01 + 1e-6)
    * 2.0 - 1.0
)

print("RAW RIGHT GRIPPER")
print("min :", x.min())
print("max :", x.max())

print("\nNORMALIZED RIGHT GRIPPER")
print("min :", x_norm.min())
print("max :", x_norm.max())
print("mean:", x_norm.mean())
print("std :", x_norm.std())

print("\nPercentiles")
for q in [
    0,
    0.01,
    0.1,
    1,
    50,
    99,
    99.9,
    100,
]:
    print(
        f"{q:7.2f}% : "
        f"{np.percentile(x_norm, q):12.4f}"
    )

print("\nCounts")
print(
    "|norm| > 10   :",
    np.sum(np.abs(x_norm) > 10)
)
print(
    "|norm| > 100  :",
    np.sum(np.abs(x_norm) > 100)
)
print(
    "|norm| > 1000 :",
    np.sum(np.abs(x_norm) > 1000)
)
print(
    "|norm| > 10000:",
    np.sum(np.abs(x_norm) > 10000)
)