from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from app_classifier.inference import AppClassifier as _AppClassifier
from app_classifier.inference import InferConfig as _InferConfig


@dataclass
class AppInferConfig:
    repo_id: str
    revision: Optional[str] = None
    hf_token: Optional[str] = None
    device: str = "cuda"
    max_length: int = 512
    batch_size: int = 8
    load_in_4bit: bool = True
    prefer_bf16: bool = True
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    label_mapping_path: Optional[str] = None
    mode: Optional[str] = None
    positive_label: Optional[str] = None


class AppRiskClassifier:
    def __init__(self, cfg: AppInferConfig):
        self.cfg = cfg
        self._classifier: Optional[_AppClassifier] = None

    def load_models(self):
        infer_cfg = _InferConfig(
            model_id=self.cfg.repo_id,
            revision=self.cfg.revision,
            hf_token=self.cfg.hf_token,
            device=self.cfg.device,
            max_length=self.cfg.max_length,
            batch_size=self.cfg.batch_size,
            load_in_4bit=self.cfg.load_in_4bit,
            prefer_bf16=self.cfg.prefer_bf16,
            base_model_name=self.cfg.base_model_name,
            label_mapping_path=self.cfg.label_mapping_path,
            mode=self.cfg.mode,
            positive_label=self.cfg.positive_label,
        )
        self._classifier = _AppClassifier(infer_cfg).load_model()
        return self

    def are_models_loaded(self) -> bool:
        return self._classifier is not None and self._classifier.are_models_loaded()

    def classify_apps_batch(
        self,
        entries: Sequence[Dict[str, Any]],
        prediction_ids: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.are_models_loaded():
            raise RuntimeError("Models not loaded. Call load_models() first.")

        preds = self._classifier.predict_records(entries)
        out = []
        for i, pred in enumerate(preds):
            prediction_id = str(prediction_ids[i]) if prediction_ids is not None else str(i)
            out.append({"prediction_id": prediction_id, **pred})
        return out
