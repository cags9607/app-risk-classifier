import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class TaskConfig:
    task_name: str
    mode: str
    model_name: str
    hf_model_id: Optional[str]
    output_dir: Optional[str]
    label_col: Optional[str]
    sample_bucket_col: str = "sample_bucket"
    title_col: str = "title"
    description_col: str = "description"
    bundle_id_col: str = "bundle_id"
    output_cols: Optional[Dict[str, str]] = None
    positive_labels: Optional[list] = None
    negative_labels: Optional[list] = None


def load_task_config(path_or_task: str) -> TaskConfig:
    candidate = Path(path_or_task)

    if not candidate.exists():
        candidate = Path("configs") / f"{path_or_task}.json"

    if not candidate.exists():
        raise FileNotFoundError(f"Could not find config: {path_or_task}")

    with open(candidate, "r", encoding = "utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    return TaskConfig(
        task_name = data.get("task_name"),
        mode = data.get("mode"),
        model_name = data.get("model_name", "Qwen/Qwen2.5-3B-Instruct"),
        hf_model_id = data.get("hf_model_id"),
        output_dir = data.get("output_dir"),
        label_col = data.get("label_col"),
        sample_bucket_col = data.get("sample_bucket_col", "sample_bucket"),
        title_col = data.get("title_col", "title"),
        description_col = data.get("description_col", "description"),
        bundle_id_col = data.get("bundle_id_col", "bundle_id"),
        output_cols = data.get("output_cols"),
        positive_labels = data.get("positive_labels"),
        negative_labels = data.get("negative_labels"),
    )
