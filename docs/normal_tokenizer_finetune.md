# Infinity Normal Tokenizer Fine-Tuning

This repository includes a dedicated normal-map tokenizer fine-tuning pipeline for Hypersim caches exported by NormalART.

## Data Format

The trainer expects NormalART cache files:

- `train.pt`
- `val.pt`

Each cache must contain:

- `targets`: `[N, 3, H, W]` normal maps in `[-1, 1]`
- `masks`: `[N, 1, H, W]` valid-pixel masks
- `metadata`: optional per-sample metadata

The default paths point to:

- `/root/vepfs/NormalART/datasets/cache/hypersim_full/256x256/train.pt`
- `/root/vepfs/NormalART/datasets/cache/hypersim_full/256x256/val.pt`

## Base Tokenizer

The fine-tuning script warm-starts from:

- `weights/infinity_vae_d56_f8_14_patchify.pth`

That configuration corresponds to:

- `codebook_dim=14`
- `apply_spatial_patchify=1`
- patch size `8`

Downstream code that loads the fine-tuned tokenizer should keep the same tokenizer shape assumptions.

## Stable Training Command

The tested stable launch on 2x A100 80G is:

```bash
cd /root/vepfs/Infinity
NPROC_PER_NODE=2 bash scripts/train_normal_tokenizer.sh
```

By default the wrapper now uses:

- `batch-size=16` per GPU
- `val-batch-size=16` per GPU
- `num-workers=8`
- full Hypersim cache from NormalART
- `swanlab-mode=cloud`
- `swanlab-project=infinity_normal_tokenizer_hypersim`
- `swanlab-experiment-name=train_normal_tokenizer_YYYY-MM-DD_HH-MM-SS`

This creates runs under:

- `outputs/YYYY-MM-DD/HH-MM-SS`

and refreshes:

- `outputs/latest_tokenizer_normal`

## Useful Overrides

Run a shorter stability check:

```bash
NPROC_PER_NODE=2 bash scripts/train_normal_tokenizer.sh --epochs 1 --max-steps 100
```

Run on a tiny cache:

```bash
NPROC_PER_NODE=2 bash scripts/train_normal_tokenizer.sh \
  --train-cache /root/vepfs/NormalART/datasets/cache/hypersim_tiny/256x256/train.pt \
  --val-cache /root/vepfs/NormalART/datasets/cache/hypersim_tiny/256x256/val.pt \
  --epochs 1 \
  --max-steps 20
```

Override defaults without editing the script:

```bash
NORMAL_TOKENIZER_BATCH_SIZE=12 \
NORMAL_TOKENIZER_VAL_BATCH_SIZE=12 \
NORMAL_TOKENIZER_NUM_WORKERS=4 \
NPROC_PER_NODE=2 \
bash scripts/train_normal_tokenizer.sh
```

Override SwanLab mode explicitly when needed:

```bash
SWANLAB_MODE=offline \
NPROC_PER_NODE=2 \
bash scripts/train_normal_tokenizer.sh
```

## Outputs

Each run writes:

- `train.log`
- `train_rankXX.log`
- `args.json`
- `swanlab/`
- `swanlab_run.json`
- `images/`
- `checkpoints/last.pth`
- `checkpoints/best_angle_*.pth`

If `checkpoints/last.pth` already exists in the selected output directory, the trainer resumes automatically.
