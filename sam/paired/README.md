# 全量配对背景增强：200原始 + 200增强

本目录是原处理脚本的独立副本，不改动正在运行的 `sam/run_process.sh`。

启动：

```bash
cd /root/data1/xxy/openpi/sam/paired
nohup bash run_pairs.sh > pipeline.log 2>&1 < /dev/null &
```

默认GPU4–7补做旧任务未选择增强的100条；旧任务继续使用GPU0–3。
补做完成后等待旧任务释放锁，复用其中已完成、校验通过且处理配置一致的增强结果。
旧任务失败时只复用已完成结果；缺少的增强episode由新流程补齐。
全部200条增强通过帧数、标签、视频校验和及异常比例检查后，再合并和同步。

中间增强数据：
`/root/data/xxy/openpi/datasets/competition/cup_four_cameras_bgaug32_sam2_all200`

最终训练数据：
`/root/data/xxy/openpi/datasets/competition/cup_four_cameras_bgaug32_sam2_pairs400`

同步副本：
`/root/data1/xxy/huggingface/lerobot/competition/cup_four_cameras_bgaug32_sam2_pairs400`

repo_id：`competition/cup_four_cameras_bgaug32_sam2_pairs400`

最终400条中，0–199是原始版本，200–399是对应的增强版本。
总帧数291956、四路视频1600个，仍然只有200条独立示范。
每对版本帧数相同，均匀按帧采样时原始/增强版本各占50%。
不安全分割帧保留原图，因此真正替换背景的帧占比会低于50%，完成报告记录此值。

动作、状态、时间戳、任务编号、episode内frame_index保持不变。
仅episode_index与全局index重排。图像统计按最终视频计算；动作/状态的原有统计保持，count翻倍。
`meta/augmentation_pairs.jsonl`记录配对关系。任何训练/验证划分都必须把同一pair_id放在同一组。
默认info.splits沿用全量训练，未添加独立验证集，也不自行启动训练或改动训练配置。

`pipeline_status.json`是整个新流程状态；`status.json`为增强子流程状态；`logs/worker_*.log`为帧级进度。
完成校验前使用`.processing`目录，不可用于训练。完整数据与同步校验后才出现完成标记。

修改续跑规则：必须同时满足增强标志、渲染配置指纹和视频校验和才能跳过，不能把已复制原片当成增强结果。
原始数据与旧处理结果均不删除、不覆盖。

`test_pairs.py`使用临时合成数据验证合并、索引、动作/状态/时间戳不变、四路视频统计、断点续跑，并通过服务器现有LeRobot数据加载器读取四路视频及50步动作chunk，确认动作chunk不会跨越episode边界。
