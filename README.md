# Infinity $\infty$：面向高分辨率图像合成的 Bitwise 自回归建模

<div align="center">

[![demo platform](https://img.shields.io/badge/Play%20with%20Infinity%21-Infinity%20demo%20platform-lightblue)](https://opensource.bytedance.com/gmpt/t2i/invite)&nbsp;
[![Project Page](https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages)](https://foundationvision.github.io/infinity.project/)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2412.04431-b31b1b.svg)](https://arxiv.org/abs/2412.04431)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-FoundationVision/Infinity-yellow)](https://huggingface.co/FoundationVision/infinity)&nbsp;
[![code](https://img.shields.io/badge/%F0%9F%A4%96%20Code-FoundationVision/Infinity-green)](https://github.com/FoundationVision/Infinity)&nbsp;
[![Replicate](https://replicate.com/chenxwh/infinity/badge)](https://replicate.com/chenxwh/infinity)&nbsp;

</div>

<p align="center" style="font-size: larger;">
  <a href="https://arxiv.org/abs/2412.04431">Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis</a>
</p>

<p align="center">
<img src="assets/show_images.jpg" width=95%>
<p>

## 更新

- 2025-11-07：发布基于 VAR 和 Infinity 的文生视频项目，见 [InfinityStar](https://github.com/FoundationVision/InfinityStar)。
- 2025-06-24：发布 Infinity-8B 生成 512x512 图像的中间阶段模型。
- 2025-05-25：发布 Infinity 图像 tokenizer 训练代码和配置，见 [BitVAE](https://github.com/FoundationVision/BitVAE)。
- 2025-04-24：Infinity 被 CVPR 2025 接收为 Oral。
- 2025-02-18：发布 Infinity-8B 权重和代码。
- 2025-02-07：发布 Infinity-8B Demo，见 [demo](https://opensource.bytedance.com/gmpt/t2i/invite)。
- 2024-12-24：发布训练/测试代码、checkpoint 和 demo。
- 2024-12-12：添加项目主页。
- 2024-12-10：Visual AutoRegressive Modeling 获得 NeurIPS 2024 Best Paper Award。
- 2024-12-05：发布论文。

## 体验 Infinity

可以通过 [demo website](https://opensource.bytedance.com/gmpt/t2i/invite) 交互式体验 Infinity 文生图能力。

仓库也提供了 [interactive_infer.ipynb](tools/interactive_infer.ipynb) 和 [interactive_infer_8b.ipynb](tools/interactive_infer_8b.ipynb)，用于查看 Infinity-2B 和 Infinity-8B 的推理细节。

本仓库额外提供一个本地实验 TUI，可以一键启动文生图推理、法线推理、法线训练、tokenizer 微调、评估和常用检查：

```bash
cd /root/vepfs/Infinity
.venv/bin/python tui.py
```

## 开源计划

- [ ] Infinity-20B Checkpoints
- [x] Infinity Image tokenizer training code & setting
- [x] Infinity-8B Checkpoints (512x512)
- [x] Infinity-8B Checkpoints (1024x1024)
- [x] Training Code
- [x] Web Demo
- [x] Inference Code
- [x] Infinity-2B Checkpoints
- [x] Visual Tokenizer Checkpoints

## 项目简介

Infinity 是一种 Bitwise Visual AutoRegressive Modeling 方法，可以生成高分辨率、写实图像。它在 bitwise token 预测框架下重新定义视觉自回归模型，引入 infinite-vocabulary tokenizer、infinite-vocabulary classifier 和 bitwise self-correction。通过在理论上把 tokenizer 词表规模扩展到无限大，并同步扩大 transformer 规模，Infinity 释放了更强的 scaling 能力。

Infinity 在自回归文生图模型中刷新了结果，并超过 SD3-Medium、SDXL 等强扩散模型。论文报告中，Infinity 将 GenEval 分数从 0.62 提升到 0.73，将 ImageReward 分数从 0.87 提升到 0.96，并获得 66% 的胜率。在没有额外优化的情况下，Infinity 可以在 0.8 秒内生成一张 1024x1024 高质量图像，比 SD3-Medium 快 2.6 倍。

### 在 bitwise token 预测框架下重定义 VAR

<p align="center">
<img src="assets/framework_row.png" width=95%>
<p>

**Infinite-Vocabulary Tokenizer**：提出 bitwise multi-scale residual quantizer，显著降低内存占用，使训练极大词表成为可能，例如 $V_d = 2^{32}$ 或 $V_d = 2^{64}$。

**Infinite-Vocabulary Classifier**：传统 classifier 预测 $2^d$ 个 index，IVC 只预测 $d$ 个 bit。连续特征中接近 0 的轻微扰动可能导致 index label 完全变化，而 bit label 的变化更平滑，监督信号更稳定。如果 d = 32 且 h = 2048，传统 classifier 需要 8.8T 参数，而 IVC 只需要 0.13M 参数。

**Bitwise Self-Correction**：自回归训练中的 teacher forcing 会带来严重的训练-测试不一致，使 transformer 更像是在细化特征，而不是识别并修正错误。错误会在生成过程中传播和放大。Bitwise Self-Correction 用于缓解这一问题。

### 扩大 Vocabulary 有利于重建和生成

<p align="center">
<img src="assets/scaling_vocabulary.png" width=95%>
<p>

### Infinity Transformer 的 Scaling Law

<p align="center">
<img src="assets/scaling_models.png" width=95%>
<p>

## Infinity Model Zoo

Infinity 模型权重可以从 <a href='https://huggingface.co/FoundationVision/infinity'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20weights-FoundationVision/Infinity-yellow'></a> 下载，也可以使用下表中的链接。

### Visual Tokenizer

Infinity 图像 tokenizer 训练代码和配置见 [BitVAE](https://github.com/FoundationVision/BitVAE)。

| vocabulary | stride | IN-256 rFID $\downarrow$ | IN-256 PSNR $\uparrow$ | IN-512 rFID $\downarrow$ | IN-512 PSNR $\uparrow$ | HF weights |
|:----------:|:------:|:------------------------:|:----------------------:|:------------------------:|:----------------------:|:----------|
| $V_d=2^{16}$ | 16 | 1.22 | 20.9 | 0.31 | 22.6 | [infinity_vae_d16.pth](https://huggingface.co/FoundationVision/infinity/blob/main/infinity_vae_d16.pth) |
| $V_d=2^{24}$ | 16 | 0.75 | 22.0 | 0.30 | 23.5 | [infinity_vae_d24.pth](https://huggingface.co/FoundationVision/infinity/blob/main/infinity_vae_d24.pth) |
| $V_d=2^{32}$ | 16 | 0.61 | 22.7 | 0.23 | 24.4 | [infinity_vae_d32.pth](https://huggingface.co/FoundationVision/infinity/blob/main/infinity_vae_d32.pth) |
| $V_d=2^{64}$ | 16 | 0.33 | 24.9 | 0.15 | 26.4 | [infinity_vae_d64.pth](https://huggingface.co/FoundationVision/infinity/blob/main/infinity_vae_d64.pth) |
| $V_d=2^{32}$ | 16 | 0.75 | 21.9 | 0.32 | 23.6 | [infinity_vae_d32_reg.pth](https://huggingface.co/FoundationVision/Infinity/blob/main/infinity_vae_d32reg.pth) |

### Infinity

| model | Resolution | GenEval | DPG | HPSv2.1 | HF weights |
|:-----:|:----------:|:-------:|:---:|:-------:|:----------|
| Infinity-2B | 1024 | 0.69 / 0.73 $^{\dagger}$ | 83.5 | 32.2 | [infinity_2b_reg.pth](https://huggingface.co/FoundationVision/infinity/blob/main/infinity_2b_reg.pth) |
| Infinity-8B | 1024 | 0.79 $^{\dagger}$ | 86.6 | - | [infinity_8b_weights](https://huggingface.co/FoundationVision/Infinity/tree/main/infinity_8b_weights) |
| Infinity-8B | 512 | - | - | - | [infinity_8b_512x512_weights](https://huggingface.co/FoundationVision/Infinity/tree/main/infinity_8b_512x512_weights) |
| Infinity-20B | 1024 | - | - | - | Coming Soon |

$^{\dagger}$ 表示使用了 [prompt rewriter](tools/prompt_rewriter.py) 测试。

可以通过 [interactive_infer.ipynb](tools/interactive_infer.ipynb) 和 [interactive_infer_8b.ipynb](tools/interactive_infer_8b.ipynb) 加载这些模型并生成图像。

## 安装

1. 训练加速使用 FlexAttention，需要 `torch>=2.5.1`。
2. 安装其他 Python 依赖：

```bash
pip3 install -r requirements.txt
```

3. 从 Hugging Face 下载权重。除了 VAE 和 transformer 权重，还需要下载 [flan-t5-xl](https://huggingface.co/google/flan-t5-xl)：

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
```

上面代码会把 `flan-t5-xl` 下载到 `~/.cache/huggingface`。

## 数据准备

训练数据目录由若干 jsonl 文件组成，文件名格式为 `[h_div_w_template]_[num_examples].jsonl`。其中：

- `h_div_w_template`：图片高宽比模板
- `num_examples`：该高宽比附近的样本数

[dataset_t2i_iterable.py](infinity/dataset/dataset_t2i_iterable.py) 支持超过 100M 样本的训练，但每个高宽比模板下的样本数量需要写在文件名中。

```text
/path/to/dataset/:
  [h_div_w_template1]_[num_examples].jsonl
  [h_div_w_template2]_[num_examples].jsonl
  [h_div_w_template3]_[num_examples].jsonl
```

每个 jsonl 文件中每一行是一个 JSON item，格式如下：

```json
{
  "image_path": "path/to/image, required",
  "h_div_w": "float value of h_div_w for the image, required",
  "long_caption": "long caption of the image, required",
  "long_caption_type": "InternVL 2.0, required",
  "text": "short caption of the image, optional",
  "short_caption_type": "user prompt, optional"
}
```

仓库提供了包含 10 张图片的 toy dataset，可以参考 [data/infinity_toy_data](data/infinity_toy_data) 准备自己的数据。

## 训练脚本

一条命令训练 Infinity-2B：

```bash
bash scripts/train.sh
```

法线图 tokenizer 微调，以及基于 NormalART/Hypersim 数据的 RGB 到法线图训练和推理，见 [docs/normal_readme.md](docs/normal_readme.md)。Tokenizer 微调的详细说明见 [docs/normal_tokenizer_finetune.md](docs/normal_tokenizer_finetune.md)。在 2x A100 80G 上稳定测试过的 tokenizer 启动方式是：

```bash
NPROC_PER_NODE=2 bash scripts/train_normal_tokenizer.sh
```

如果需要训练不同模型规模 `{125M, 1B, 2B}` 和不同分辨率 `{256, 512, 1024}`，可以参考下面命令：

```bash
# 125M, layer12, pixel number = 256 x 256 = 0.06M Pixels
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --model=layer12c4 --pn 0.06M --exp_name=infinity_125M_pn_0.06M

# 1B, layer24, pixel number = 256 x 256 = 0.06M Pixels
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --model=layer24c4 --pn 0.06M --exp_name=infinity_1B_pn_0.06M

# 2B, layer32, pixel number = 256 x 256 = 0.06M Pixels
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --model=2bc8 --pn 0.06M --exp_name=infinity_2B_pn_0.06M

# 2B, layer32, pixel number = 512 x 512 = 0.25M Pixels
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --model=2bc8 --pn 0.25M --exp_name=infinity_2B_pn_0.25M

# 2B, layer32, pixel number = 1024 x 1024 = 1M Pixels
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --model=2bc8 --pn 1M --exp_name=infinity_2B_pn_1M
```

训练会创建 `local_output` 目录保存 checkpoint 和日志。可以通过 `local_output/log.txt` 和 `local_output/stdout.txt` 查看训练过程。建议使用 [wandb](https://wandb.ai/site/) 记录更完整的训练日志。

如果实验中断，重新运行同一命令即可，训练会从 `local_output/ckpt*.pth` 中的最新 checkpoint **自动恢复**。

## 评估

一条命令运行评估：

```bash
bash scripts/eval.sh
```

[eval.sh](scripts/eval.sh) 支持常用评估指标，包括 [GenEval](https://github.com/djghosh13/geneval)、[ImageReward](https://github.com/THUDM/ImageReward)、[HPSv2.1](https://github.com/tgxs002/HPSv2)、FID 和 Validation Loss。更多说明见 [evaluation/README.md](evaluation/README.md)。

## 微调

微调 Infinity 时，在 [train.sh](scripts/train.sh) 的训练命令中追加 `--rush_resume=[infinity_2b_reg.pth]` 即可。需要特别注意 `--pn`，它决定训练和推理的图像分辨率：

```text
--pn=0.06M  # 256x256 分辨率，也包括相同像素数量的其他宽高比
--pn=0.25M  # 512x512 分辨率
--pn=1M     # 1024x1024 分辨率
```

微调后会得到类似 `[model_dir]/ar-ckpt-giter(xxx)K-ep(xxx)-iter(xxx)-last.pth` 的 checkpoint。该 checkpoint 不只包含模型权重，也包含训练状态。用这个模型推理时，需要在 [eval.sh](scripts/eval.sh) 或 [interactive_infer.ipynb](tools/interactive_infer.ipynb) 中启用 `--enable_model_cache=1`。

## Docker 使用

如果只想在本地复现论文模型的推理，可以使用 Docker 容器。这种方式适合不想手动配置环境的用户。

### 1. 下载权重

将 `flan-t5-xl` 文件夹、`infinity_2b_reg.pth` 和 `infinity_vae_d32reg.pth` 下载到 `weights` 目录。

### 2. 构建 Docker 容器

```bash
docker build -t my-flash-attn-env .
docker run --gpus all -it --name my-container -v {your-local-path}:/workspace my-flash-attn-env
```

### 3. 运行

```bash
python Infinity/tools/reproduce.py
```

也可以修改 `reproduce.py` 中的 prompt，使用自己的提示词生成图片。

## Infinity-8B 与 Infinity-2B 对比

Infinity 展现了较强的 scaling 能力，因此进一步扩展到了更大的模型规模。下表展示 Infinity-2B 和 Infinity-8B 的并排对比结果。

| Prompt | Infinity (# params=2B) | Infinity (# params=8B) |
|:-------|:----------------------:|:----------------------:|
| a cat holds a sign with the text 'Diffusion is dead' | ![](assets/2b_8b/1l.webp) | ![](assets/2b_8b/1r.webp) |
| A beautiful Chinese woman with graceful features, close-up portrait, long flowing black hair, wearing a traditional silk cheongsam delicately embroidered with floral patterns, face softly illuminated by ambient light, serene expression | ![](assets/2b_8b/2l.webp) | ![](assets/2b_8b/2r.webp) |
| a Chinese model is sitting on a train, magazine cover, clothes made of plastic, photorealistic, futuristic style, gray and green light, movie lighting, 32K HD | ![](assets/2b_8b/3l.webp) | ![](assets/2b_8b/3r.webp) |
| A group of students in a class | ![](assets/2b_20b/4l.jpg) | ![](assets/2b_8b/4r.webp) |

## 引用

如果本项目对你的研究有帮助，欢迎 star 或引用：

```bibtex
@misc{Infinity,
    title={Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis},
    author={Jian Han and Jinlai Liu and Yi Jiang and Bin Yan and Yuqi Zhang and Zehuan Yuan and Bingyue Peng and Xiaobing Liu},
    year={2024},
    eprint={2412.04431},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2412.04431},
}
```

```bibtex
@misc{VAR,
    title={Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction},
    author={Keyu Tian and Yi Jiang and Zehuan Yuan and Bingyue Peng and Liwei Wang},
    year={2024},
    eprint={2404.02905},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2404.02905},
}
```

## License

本项目使用 MIT License，详情见 [LICENSE](LICENSE)。
