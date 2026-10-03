# 独立四路背景增强数据处理工具

源数据：`/root/data/xxy/openpi/datasets/competition/cup_four_cameras`

最终输出：`/root/data/xxy/openpi/datasets/competition/cup_four_cameras_bgaug32_sam2`

同步副本：`/root/data1/xxy/huggingface/lerobot/competition/cup_four_cameras_bgaug32_sam2`

## 启动

```bash
cd /root/data1/xxy/openpi/sam
nohup bash run_process.sh > processing.log 2>&1 < /dev/null &
```

默认使用GPU 0、1、2、3；可在启动前用 `SAM_GPUS` 指定空闲显卡。环境复用已有 CUDA torch 只读依赖，不升级现有 OpenPI 环境。环境和大模型文件实际保存在本地磁盘 `/root/data/xxy/openpi/sam_runtime`，`.venv` 是该环境的链接。

## 算法与范围

每条增强视频逐帧执行本地 Grounding DINO目标检测，再由SAM 2.1分割保护对象。不依赖固定腕部遮罩或预设动作时间；这版不是首帧提示的长视频传播，避免遗漏中途出现的杯子和另一只手。固定目标图仅在像素完全相同时复用分割。

保护对象：机器人手、机械臂、纸杯、取杯装置、封口机、打印机、相关桌面。12像素保护外扩，外侧6像素过渡。只替换外围背景，不另加全图颜色、裁剪或旋转。

固定随机种子选择100条episode增强，另100条视频保持原始字节，以近似50%原图/50%增强的训练混合，并避免同一视频逐帧切换背景产生闪烁。每条episode四路使用协调、全程不变的背景风格；动作、状态、索引、任务、时间戳、episode数和帧数保持不变。

如果未检出必要对象、前景面积异常、蓝紫色杯/手疑似漏保护或面积跳变，则保留原始帧并记录。统计检查不能替代视觉检查，无法保证检测模型不存在漏检。

## 完整性与同步

先生成 `.processing` 数据目录，不让不完整数据看起来已经可训练。每个视频验证帧数，逐帧验证保护范围的RGB在编码前不变；视频使用标准H264/yuv420p重新编码，因此编码后像素不保证逐位一致。原始episode视频不重新编码。

所有非图像Parquet逐文件SHA256保持一致。图像统计按最终解码视频重新计算；非图像统计保持原样。全部完成后重命名最终目录，再使用rsync同步并验证文件校验和。过多异常会停在暂存目录并生成质量报告，不自动声称训练就绪。

`status.json`：总体状态；`logs/worker_*.log`：进度；新数据中的`processing_reports/`：检查预览、异常和episode报告。`PROCESSING_COMPLETE.json`：完整校验清单。

停止或失败后重新运行同一脚本可复核并跳过已完成episode。不会覆盖原始数据，不使用rsync删除操作，不启动模型训练。

本地工具无按视频收费API；使用服务器算力和存储。未导出服务器视频到本地。
