# Normal Tokenizer Fine-Tuning

Use `tools/train_normal_tokenizer.py` to fine-tune the normal-map tokenizer.

## Recommended Launch

From the repository root:

```bash
NPROC_PER_NODE=8 bash scripts/train_normal_tokenizer.sh
```

The script creates a managed run directory under `outputs/` unless `--output-dir` is provided.

## Dataset Mix

The tokenizer uses the same normal dataset mixing controls as RGB-to-normal estimation:

```bash
--train-datasets hypersim,vkitti2
--train-dataset-weights hypersim:3,vkitti2:1
--data-root data/hypersim/processed/hypersim
--vkitti2-root data/VKITTI2
```

The `3:1` default intentionally down-weights VKITTI2 because it contains repeated scene variants. If the goal is to
optimize indoor Hypersim validation, use `--train-datasets hypersim`. If the goal is outdoor robustness, experiment
with `hypersim:2,vkitti2:1` or `hypersim:1,vkitti2:1` and add a separate VKITTI2 validation split.

## TUI Defaults

The `训练法线 Tokenizer` TUI task uses:

- `train_datasets = hypersim,vkitti2`
- `train_dataset_weights = hypersim:3,vkitti2:1`
- `vkitti2_root = data/VKITTI2`

## Notes

- Training batches are grouped by dataset and target size.
- In DDP, all ranks in a global step receive the same dataset/target-size group.
- Validation remains Hypersim by default.
- SwanLab is imported lazily and only when logging is enabled.

## Smoke Tests

```bash
scripts/check.sh
```
