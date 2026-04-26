from __future__ import annotations

import json
from pathlib import Path

from common import RAW_DIR


def file_size_mb(path: Path) -> str:
	return f"{path.stat().st_size / (1024 * 1024):.2f} MB"


def inspect_json_sample(path: Path) -> str:
	try:
		with path.open("r", encoding="utf-8") as f:
			data = json.load(f)
		if isinstance(data, list) and data:
			item = data[0]
			if isinstance(item, dict):
				return f"list[{len(data)}], keys={sorted(item.keys())}"
			return f"list[{len(data)}], first_type={type(item).__name__}"
		return f"type={type(data).__name__}"
	except Exception as exc:  # pragma: no cover - best effort probe
		return f"json parse failed: {exc}"


def inspect() -> None:
	checks = {
		"finqa": [RAW_DIR / "finqa/FinQA_repo/dataset/train.json"],
		"tatqa": [RAW_DIR / "tatqa/TAT-QA_repo/dataset_raw/tatqa_dataset_train.json"],
		"ectsum": [RAW_DIR / "ectsum/ECTSum_repo/data/final/train/ects"],
		"finer": [RAW_DIR / "finer/finer-ord_repo/train.csv"],
		"finred": [RAW_DIR / "finred/train.sent", RAW_DIR / "finred/train.tup"],
		"fiqa": [RAW_DIR / "fiqa/fiqa_repo/data/train-00000-of-00001-aeefa1eadf5be10b.parquet"],
		"finsen": [RAW_DIR / "finsen/sentiment labelled sentences/amazon_cells_labelled.txt"],
	}

	print(f"RAW_DIR={RAW_DIR}")
	for name, paths in checks.items():
		print(f"\n[{name}]")
		for path in paths:
			exists = path.exists()
			print(f"- {path} | exists={exists}")
			if not exists:
				continue
			if path.is_file():
				print(f"  size={file_size_mb(path)}")
			if path.suffix == ".json":
				print(f"  sample={inspect_json_sample(path)}")
			if path.is_dir():
				sample_files = sorted([p for p in path.glob("*.txt")])[:3]
				for sf in sample_files:
					print(f"  sample_file={sf.name}")


if __name__ == "__main__":
	inspect()
