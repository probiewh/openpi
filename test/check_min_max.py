import glob
import numpy as np
import pyarrow.parquet as pq

ROOT = "/root/data1/xxy/huggingface/lerobot/g1/rubbish_70"

files = sorted(
    glob.glob(ROOT + "/**/*.parquet", recursive=True)
)

print("Parquet files:", len(files))

all_actions = []
all_states = []

for f in files:
    table = pq.read_table(f)

    if "action" in table.column_names:
        actions = np.asarray(
            table["action"].to_pylist(),
            dtype=np.float64
        )
        all_actions.append(actions)

    if "observation.state" in table.column_names:
        states = np.asarray(
            table["observation.state"].to_pylist(),
            dtype=np.float64
        )
        all_states.append(states)

actions = np.concatenate(all_actions, axis=0)
states = np.concatenate(all_states, axis=0)

print("actions shape:", actions.shape)
print("states shape :", states.shape)

def print_stats(name, x):
    print("\n" + "=" * 90)
    print(name)
    print("=" * 90)

    print("num :", len(x))
    print("min :", x.min())
    print("max :", x.max())
    print("mean:", x.mean())
    print("std :", x.std())

    for q in [
        0,
        0.001,
        0.01,
        0.1,
        1,
        50,
        99,
        99.9,
        99.99,
        100,
    ]:
        print(
            f"{q:8.3f}% : "
            f"{np.percentile(x, q):.10f}"
        )

print_stats(
    "PARQUET RIGHT GRIPPER STATE [15]",
    states[:, 15]
)

print_stats(
    "PARQUET RIGHT GRIPPER ACTION [15]",
    actions[:, 15]
)