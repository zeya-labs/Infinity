import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Remap prompt rewrite cache keys from a reference cache.")
    parser.add_argument("--reference-cache", type=Path, default=root / "prompt_rewrite_cache_1.json")
    parser.add_argument("--source-cache", type=Path, default=root / "prompt_rewrite_cache_123.json")
    parser.add_argument("--output", type=Path, default=root / "prompt_rewrite_cache.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.reference_cache.open("r", encoding="utf-8") as f:
        correct = json.load(f)
    with args.source_cache.open("r", encoding="utf-8") as f:
        false_key_dict = json.load(f)

    keys1_list = list(correct.keys())
    keys2_list = list(false_key_dict.keys())
    if len(keys1_list) != len(keys2_list):
        raise ValueError(
            f"cache sizes differ: reference={len(keys1_list)}, source={len(keys2_list)}"
        )

    final_dict = {}
    for key1, key2 in zip(keys1_list, keys2_list):
        final_dict[key1] = false_key_dict[key2]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
