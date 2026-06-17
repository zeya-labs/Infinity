# Normal Training

This repository supports two normal-related training jobs:

- normal tokenizer fine-tuning with `tools/train_normal_tokenizer.py`
- RGB-to-normal estimation with `tools/train_normal_estimation.py`

Both jobs can train on a mixture of Hypersim and VKITTI2. The default training mix is:

```bash
--train-datasets hypersim,vkitti2
--train-dataset-weights hypersim:9,vkitti2:1
--vkitti2-root data/VKITTI2
```

`--data-root` remains the Hypersim root. `--vkitti2-root` points to the VKITTI2 root containing
`processed/normals_d2nt_v3/manifest.jsonl`.

## Sampling Semantics

The mixed normal dataloader uses `GroupedTargetSizeBatchSampler`.

- A local batch contains only one dataset and one target resolution.
- In distributed training, all ranks in the same global step receive the same dataset/target-size group.
- `--train-dataset-weights` is a positive-integer step ratio.
- With `hypersim:9,vkitti2:1`, the global step pattern is `Hypersim x9, VKITTI2, ...`.

This avoids cross-rank shape skew and keeps per-step metrics interpretable.

## Default 1M Shapes

For `pn=1M`, representative target shapes are:

- Hypersim: `[3, 864, 1152]`
- VKITTI2: `[3, 592, 1776]`

The first dimension becomes the per-rank batch size during training.

## Entrypoints

TUI tasks:

- `训练 RGB 到 Normal`
- `训练法线 Tokenizer`

Shell scripts:

```bash
bash scripts/train_normal.sh
bash scripts/train_normal_tokenizer.sh
```

Environment overrides shared by both scripts:

```bash
NORMAL_TRAIN_DATASETS=hypersim,vkitti2
NORMAL_TRAIN_DATASET_WEIGHTS=hypersim:9,vkitti2:1
NORMAL_VKITTI2_ROOT=data/VKITTI2
NORMAL_DATA_ROOT=data/hypersim/processed/hypersim
```

Use `NORMAL_TRAIN_DATASETS=hypersim` to train on Hypersim only.

## Validation

The current normal tokenizer and RGB-to-normal training defaults validate on Hypersim. Keep this in mind when
interpreting metrics after adding VKITTI2 to training: VKITTI2 improves outdoor coverage but is not reflected in the
default validation split unless a separate VKITTI2 validation workflow is added.

## Upload Checkpoints to Hugging Face

Install the Hub client and authenticate without writing tokens into the repository:

```bash
python -m pip install huggingface_hub
export HF_TOKEN=<your-token>
```

Upload a normal-estimation checkpoint:

```bash
python scripts/upload_hf_checkpoint.py \
  --repo-id <user-or-org>/<repo-name> \
  --checkpoint outputs/normal_estimation/<run>/checkpoints/<checkpoint>.pth \
  --path-in-repo checkpoints/<checkpoint>.pth
```

Use `--dry-run` first to confirm the target repository path. Add `--create-repo --private` when the model repository
should be created as a private Hub repository before uploading.

## Regression Tests

Run the lightweight tests with:

```bash
scripts/check.sh
```

The checks cover mixed normal sampling, SwanLab utility behavior, TUI command construction, Python syntax checks, and
normal shell entrypoint syntax checks.
