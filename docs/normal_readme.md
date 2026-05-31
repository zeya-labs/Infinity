# Infinity 法线相关流程说明

这份文档汇总当前仓库里的法线图扩展能力：

- 法线图 tokenizer 微调
- RGB 到法线图的训练
- RGB 到法线图的推理

这套流程主要面向 NormalART 导出或缓存的 Hypersim 数据，使用 Infinity BSQ-VAE tokenizer 和 Infinity 自回归主干模型。

## 相关文件

- `docs/normal_tokenizer_finetune.md`：法线 tokenizer 微调的详细说明
- `scripts/train_normal_tokenizer.sh`：法线 tokenizer 微调启动脚本
- `tools/train_normal_tokenizer.py`：法线 tokenizer 训练入口
- `scripts/train_normal.sh`：RGB 到法线图训练启动脚本
- `tools/train_normal_estimation.py`：RGB 到法线图训练入口
- `scripts/infer_normal.sh`：RGB 到法线图推理启动脚本
- `tools/run_normal_estimation.py`：RGB 到法线图推理入口
- `infinity/normal_estimation/`：数据读取、模型组装、文件 IO 和法线工具函数

## 所需资源

默认脚本会使用这些本地权重：

```text
weights/infinity_8b_weights
weights/infinity_vae_d56_f8_14_patchify.pth
```

法线 tokenizer 微调脚本默认读取 NormalART cache：

```text
/root/vepfs/NormalART/datasets/cache/hypersim_full/256x256/train.pt
/root/vepfs/NormalART/datasets/cache/hypersim_full/256x256/val.pt
```

RGB 到法线图训练脚本默认读取处理后的 Hypersim 数据：

```text
/root/vepfs/NormalART/datasets/processed/hypersim
```

如果你的路径不同，可以用环境变量或命令行参数覆盖默认值。

## 法线 Tokenizer 微调

当你需要把 Infinity 图像 tokenizer 适配到 normal map 时，使用这条链路。

```bash
cd /root/vepfs/Infinity
NPROC_PER_NODE=2 bash scripts/train_normal_tokenizer.sh
```

常用环境变量覆盖：

```bash
NORMAL_TRAIN_CACHE=/path/to/train.pt \
NORMAL_VAL_CACHE=/path/to/val.pt \
NORMAL_TOKENIZER_BATCH_SIZE=12 \
NORMAL_TOKENIZER_VAL_BATCH_SIZE=12 \
NPROC_PER_NODE=2 \
bash scripts/train_normal_tokenizer.sh
```

tokenizer 训练支持三种可训练范围：

```bash
NPROC_PER_NODE=2 bash scripts/train_normal_tokenizer.sh \
  --trainable-scope decoder_only
```

可选范围：

- `all`：训练 encoder、quantizer 和 decoder
- `decoder_quantizer`：冻结 encoder，只训练 quantizer 和 decoder
- `decoder_only`：冻结 encoder 和 quantizer，只训练 decoder

输出目录：

```text
outputs/YYYY-MM-DD/HH-MM-SS
outputs/latest_tokenizer_normal -> 最新一次 tokenizer 训练
```

更细的 tokenizer 说明见 `docs/normal_tokenizer_finetune.md`。

## RGB 到法线图训练

当你需要训练 Infinity 主干，让模型根据 RGB 图像 token 预测法线图 token 时，使用这条链路。

```bash
cd /root/vepfs/Infinity
NPROC_PER_NODE=2 bash scripts/train_normal.sh
```

启动脚本默认创建托管输出目录：

```text
outputs/YYYY-MM-DD/HH-MM-SS
outputs/latest_normal_estimation -> 最新一次 normal estimation 训练
```

常用环境变量覆盖：

```bash
NORMAL_DATA_ROOT=/path/to/processed/hypersim \
NORMAL_VAE_CKPT=/path/to/normal_vae.pth \
RGB_VAE_CKPT=/path/to/rgb_vae.pth \
INIT_MODEL_PATH=/path/to/infinity_8b_weights \
NORMAL_BATCH_SIZE=4 \
NORMAL_VAL_BATCH_SIZE=4 \
NORMAL_ZERO=3 \
NPROC_PER_NODE=2 \
bash scripts/train_normal.sh
```

常用训练参数：

```bash
NPROC_PER_NODE=2 bash scripts/train_normal.sh \
  --epochs 20 \
  --pn 0.06M \
  --precision bf16 \
  --image-log-every 200 \
  --swanlab-mode cloud
```

分辨率由 `--pn` 控制：

- `0.06M`：约 256x256
- `0.25M`：约 512x512
- `1M`：约 1024x1024

多卡训练 Infinity-8B 时，启动脚本默认使用 `--zero 3`。如果显存足够并且不想用 FSDP，可以设置 `NORMAL_ZERO=0`。

## 日志和 Checkpoint

训练会写出：

```text
args.json
train.log
train_rankXX.log
images/
checkpoints/last.pth
checkpoints/best_angle_*.pth
swanlab/
swanlab_run.json
```

normal estimation 训练会记录训练和验证指标：

- 总 loss
- cross-entropy loss
- 法线辅助 loss
- 平均角度误差
- 11.25、22.5、30 度阈值内准确率
- learning rate

可视化输出包括：

- RGB 输入图
- 目标法线图
- 预测法线图
- 角度误差图
- 对比网格图

如果不需要 SwanLab，使用：

```bash
--swanlab-mode disabled
```

## RGB 到法线图推理

训练后可以直接用最新 checkpoint 推理：

```bash
cd /root/vepfs/Infinity
python tools/run_normal_estimation.py \
  --model-path outputs/latest_normal_estimation/checkpoints/last.pth \
  --input-path /path/to/image_or_folder \
  --output-dir outputs/normal_predictions
```

也可以使用包装脚本。它默认读取 `outputs/latest_normal_estimation/checkpoints/last.pth`，并把结果写到 `outputs/normal_predictions`：

```bash
bash scripts/infer_normal.sh \
  --input-path /path/to/image_or_folder
```

保存原始法线数组 `.npy`：

```bash
bash scripts/infer_normal.sh \
  --input-path /path/to/image_or_folder \
  --save-npy
```

推理脚本会优先从 checkpoint 的 `args` 字段读取模型配置。需要时可以显式覆盖参数：

```bash
bash scripts/infer_normal.sh \
  --input-path /path/to/images \
  --pn 0.06M \
  --tau 1.0 \
  --top-k 1
```

## 快速自检

tokenizer 短跑检查：

```bash
NPROC_PER_NODE=1 bash scripts/train_normal_tokenizer.sh \
  --epochs 1 \
  --max-steps 20 \
  --swanlab-mode disabled
```

normal estimation 短跑检查：

```bash
NPROC_PER_NODE=1 NORMAL_ZERO=0 bash scripts/train_normal.sh \
  --epochs 1 \
  --max-steps 20 \
  --swanlab-mode disabled
```

这些命令仍然需要有效的数据路径和权重路径。
