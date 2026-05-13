from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import json
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel

from .text import merge_title_description


@dataclass
class InferConfig:
    model_id: str
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


def _pick_dtype(prefer_bf16: bool = True):
    if torch.cuda.is_available() and prefer_bf16 and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _safe_prob_col(label: str) -> str:
    s = str(label).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "label"


def _load_label_mapping(path: Optional[str]) -> Optional[Dict[str, Dict[str, Any]]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


class AppClassifier:
    def __init__(self, cfg: InferConfig):
        self.cfg = cfg
        self.tokenizer = None
        self.model = None
        self.id2label: Dict[int, str] = {}
        self.label2id: Dict[str, int] = {}
        self.loaded = False

    def load_model(self):
        dtype = _pick_dtype(self.cfg.prefer_bf16)

        mapping = _load_label_mapping(self.cfg.label_mapping_path)

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_id,
            revision=self.cfg.revision,
            token=self.cfg.hf_token,
            use_fast=True,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        qconfig = None
        model_kwargs = {
            "revision": self.cfg.revision,
            "token": self.cfg.hf_token,
            "trust_remote_code": True,
        }

        if self.cfg.load_in_4bit:
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
            model_kwargs["quantization_config"] = qconfig
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = dtype

        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.cfg.model_id,
                **model_kwargs,
            )
        except Exception:
            base = AutoModelForSequenceClassification.from_pretrained(
                self.cfg.base_model_name,
                num_labels=(len(mapping["label2id"]) if mapping else 2),
                id2label=({int(k): v for k, v in mapping["id2label"].items()} if mapping else None),
                label2id=(mapping["label2id"] if mapping else None),
                **model_kwargs,
            )
            model = PeftModel.from_pretrained(
                base,
                self.cfg.model_id,
                revision=self.cfg.revision,
                token=self.cfg.hf_token,
                is_trainable=False,
            )

        if hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.pad_token_id

        if not self.cfg.load_in_4bit:
            device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
            model.to(device)

        model.eval()

        if mapping:
            self.label2id = {str(k): int(v) for k, v in mapping["label2id"].items()}
            self.id2label = {int(k): str(v) for k, v in mapping["id2label"].items()}
        else:
            raw = getattr(model.config, "id2label", {}) or {}
            self.id2label = {int(k): str(v) for k, v in raw.items()}
            self.label2id = {v: k for k, v in self.id2label.items()}

        self.tokenizer = tokenizer
        self.model = model
        self.loaded = True
        return self

    def are_models_loaded(self) -> bool:
        return self.loaded and self.model is not None and self.tokenizer is not None

    @property
    def device(self):
        if self.model is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            return next(self.model.parameters()).device
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _predict_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not self.are_models_loaded():
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        probs_all = []
        for start in range(0, len(texts), int(self.cfg.batch_size)):
            chunk = list(texts[start:start + int(self.cfg.batch_size)])
            enc = self.tokenizer(
                chunk,
                truncation=True,
                max_length=int(self.cfg.max_length),
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**enc)
                probs = torch.softmax(out.logits, dim=1).detach().cpu().numpy()
            probs_all.append(probs)

        return np.vstack(probs_all) if probs_all else np.empty((0, len(self.id2label)))

    def predict_records(self, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        texts = [merge_title_description(r.get("title"), r.get("description")) for r in records]
        probs = self._predict_texts(texts)
        pred_ids = probs.argmax(axis=1) if len(probs) else np.array([], dtype=int)

        out = []
        for i, pred_id in enumerate(pred_ids):
            row = {
                "pred_label_id": int(pred_id),
                "pred_label": self.id2label.get(int(pred_id), f"LABEL_{int(pred_id)}"),
                "pred_confidence": float(probs[i, pred_id]),
            }
            for class_id in range(probs.shape[1]):
                label = self.id2label.get(int(class_id), f"LABEL_{int(class_id)}")
                row[f"pred_prob_{_safe_prob_col(label)}"] = float(probs[i, class_id])

            if probs.shape[1] == 2:
                positive_id = 1
                if self.cfg.positive_label and self.cfg.positive_label in self.label2id:
                    positive_id = int(self.label2id[self.cfg.positive_label])
                row["pred_prob_positive"] = float(probs[i, positive_id])
                row["pred_label_bin"] = int(pred_id == positive_id)

            out.append(row)

        return out

    def predict_df(
        self,
        df: pd.DataFrame,
        *,
        title_col: str = "title",
        description_col: str = "description",
    ) -> pd.DataFrame:
        records = [
            {"title": row.get(title_col), "description": row.get(description_col)}
            for row in df.to_dict(orient="records")
        ]
        pred_df = pd.DataFrame(self.predict_records(records))
        return pd.concat([df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

    def predict_one(self, *, title: str, description: str) -> Dict[str, Any]:
        return self.predict_records([{"title": title, "description": description}])[0]
