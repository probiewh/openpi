import glob
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np


ROOT = "/root/data1/xxy/huggingface/lerobot/g1/bottle"


files = sorted(
    glob.glob(
        ROOT + "/**/*.parquet",
        recursive=True
    )
)


for f in files:

    table = pq.read_table(f)

    data = table.to_pydict()

    actions = np.asarray(
        data["action"],
        dtype=np.float32
    )


    # =========================
    # 修复右夹爪 action
    # =========================

    actions[:,15] = 5.4


    data["action"] = [
        a.tolist()
        for a in actions
    ]


    new_table = pa.Table.from_pydict(
        data
    )


    pq.write_table(
        new_table,
        f
    )


    print("fixed:",f)


print("Done")