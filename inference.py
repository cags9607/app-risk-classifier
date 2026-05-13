from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import json


@dataclass
class TaskConfig:
    task_name: str
    mode: str
    model_name: str
    hf_model_id: str
    output_dir: str
    label_col: str
    sample_bucket_col: str = "sample_bucket"
    title_col: str = "title"
    description_col: str = "description"
    bundle_id_col: str = "bundle_id"
    training_features: List[str] = field(default_factory=lambda: ["title", "description"])
    labeling_features: List[str] = field(default_factory=lambda: ["bundle_id", "title", "description"])
    output_cols: Dict[str, str] = field(default_factory=dict)
    positive_labels: Optional[List[str]] = None
    negative_labels: Optional[List[str]] = None
    label_mapping_path: Optional[str] = None


def load_task_config(path: str | Path) -> TaskConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return TaskConfig(**data)


def load_label_mapping(path: str | Path) -> Dict[str, Dict[str, int | str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
