from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from infinity.dataset import dataset_t2i_iterable
from infinity.dataset.dataset_t2i_iterable import T2IIterableDataset


class T2IIterableDatasetPreprocessTest(unittest.TestCase):
    def _write_meta_file(self, root: Path, sample_count: int = 4) -> None:
        lines = [
            '{"image_path": "missing.png", "h_div_w": 1.0, "text": "short", "long_caption": "long"}\n'
            for _ in range(sample_count)
        ]
        (root / f"1.000_{sample_count}.jsonl").write_text("".join(lines), encoding="utf-8")

    def test_rank0_splits_missing_parts_without_initialized_distributed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_meta_file(root)

            dataset = T2IIterableDataset(
                meta_folder=str(root),
                pn="0.06M",
                batch_size=1,
                num_replicas=2,
                rank=0,
                dataloader_workers=1,
                buffersize=1,
            )

            part_filepaths = next(iter(dataset.h_div_w_template2generator.values()))["part_filepaths"]
            self.assertEqual(len(part_filepaths), 2)
            for path in part_filepaths:
                self.assertTrue(Path(path).exists())

    def test_split_failure_is_raised_instead_of_deferred(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_meta_file(root)

            with mock.patch.object(
                dataset_t2i_iterable,
                "split_large_txt_files",
                side_effect=ValueError("bad split"),
            ):
                with self.assertRaisesRegex(RuntimeError, "split_meta_files failed.*bad split"):
                    T2IIterableDataset(
                        meta_folder=str(root),
                        pn="0.06M",
                        batch_size=1,
                        num_replicas=2,
                        rank=0,
                        dataloader_workers=1,
                        buffersize=1,
                    )


if __name__ == "__main__":
    unittest.main()
