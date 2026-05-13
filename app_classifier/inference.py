import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .text import make_text


def _get_hf_token(token: Optional[str] = None) -> Optional[str]:
    return (
        token
        or os.getenv("APP_CLASSIFIER_HF_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )


def _load_label_mapping(
    model_id: str,
    subfolder: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    path = hf_hub_download(
        repo_id = model_id,
        filename = "label_mapping.json",
        subfolder = subfolder,
        token = token,
    )

    with open(path, "r", encoding = "utf-8") as f:
        mapping = json.load(f)

    if "label2id" not in mapping or "id2label" not in mapping:
        raise ValueError("label_mapping.json must contain label2id and id2label.")

    label2id = {
        str(label): int(label_id)
        for label, label_id in mapping["label2id"].items()
    }

    id2label = {
        int(label_id): str(label)
        for label_id, label in mapping["id2label"].items()
    }

    expected_ids = set(range(len(id2label)))
    observed_ids = set(id2label.keys())

    if observed_ids != expected_ids:
        raise ValueError(
            "id2label ids must be contiguous integers starting at 0. "
            f"Observed ids: {sorted(observed_ids)}"
        )

    for label, label_id in label2id.items():
        if id2label.get(label_id) != label:
            raise ValueError(
                "label2id and id2label are inconsistent for "
                f"label={label!r}, id={label_id!r}."
            )

    return {
        "label2id": label2id,
        "id2label": id2label,
    }


def _safe_prob_column_name(label: str) -> str:
    return (
        str(label)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


class AppRiskClassifier:
    def __init__(
        self,
        model_id: str,
        subfolder: Optional[str] = None,
        token: Optional[str] = None,
        base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        max_length: int = 512,
        device_map: str = "auto",
        load_in_4bit: bool = True,
    ):
        self.model_id = model_id
        self.subfolder = subfolder
        self.token = _get_hf_token(token)
        self.base_model_name = base_model_name
        self.max_length = max_length

        self.label_mapping = _load_label_mapping(
            model_id = model_id,
            subfolder = subfolder,
            token = self.token,
        )

        self.label2id = self.label_mapping["label2id"]
        self.id2label = self.label_mapping["id2label"]
        self.num_labels = len(self.id2label)

        # Use fp16 for inference. bf16 logits can fail when converting to NumPy.
        compute_dtype = torch.float16

        quantization_config = None

        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_quant_type = "nf4",
                bnb_4bit_use_double_quant = True,
                bnb_4bit_compute_dtype = compute_dtype,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            subfolder = subfolder,
            token = self.token,
            use_fast = True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels = self.num_labels,
            id2label = {
                int(label_id): label
                for label_id, label in self.id2label.items()
            },
            label2id = {
                label: int(label_id)
                for label, label_id in self.label2id.items()
            },
            quantization_config = quantization_config,
            device_map = device_map,
            dtype = compute_dtype,
            token = self.token,
        )

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.problem_type = "single_label_classification"

        self.model = PeftModel.from_pretrained(
            self.model,
            model_id,
            subfolder = subfolder,
            token = self.token,
        )

        self.model.eval()

    @classmethod
    def from_hf(
        cls,
        model_id: str,
        subfolder: Optional[str] = None,
        token: Optional[str] = None,
        **kwargs,
    ):
        return cls(
            model_id = model_id,
            subfolder = subfolder,
            token = token,
            **kwargs,
        )

    def predict_texts(
        self,
        texts: List[str],
        batch_size: int = 8,
    ) -> pd.DataFrame:
        rows = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]

            enc = self.tokenizer(
                batch_texts,
                truncation = True,
                max_length = self.max_length,
                padding = True,
                return_tensors = "pt",
            )

            enc = {
                k: v.to(self.model.device)
                for k, v in enc.items()
            }

            with torch.no_grad():
                out = self.model(**enc)

                # Important:
                # Some environments return bf16/fp16 logits. Convert logits to
                # fp32 before softmax and before moving to NumPy.
                logits = out.logits.float()
                probs = torch.softmax(logits, dim = 1).detach().cpu().numpy()

            pred_ids = probs.argmax(axis = 1)
            pred_conf = probs.max(axis = 1)

            for i, pred_id in enumerate(pred_ids):
                pred_id = int(pred_id)

                row = {
                    "pred_label_id": pred_id,
                    "pred_label": self.id2label[pred_id],
                    "pred_confidence": float(pred_conf[i]),
                }

                for class_id, class_name in self.id2label.items():
                    class_id = int(class_id)
                    safe_name = _safe_prob_column_name(class_name)

                    row[f"pred_prob_{safe_name}"] = float(probs[i, class_id])

                rows.append(row)

        return pd.DataFrame(rows)

    def predict_df(
        self,
        df: pd.DataFrame,
        title_col: str = "title",
        description_col: str = "description",
        batch_size: int = 8,
    ) -> pd.DataFrame:
        if title_col not in df.columns:
            raise ValueError(f"Missing title column: {title_col}")

        if description_col not in df.columns:
            raise ValueError(f"Missing description column: {description_col}")

        texts = [
            make_text(title, description)
            for title, description in zip(df[title_col], df[description_col])
        ]

        pred_df = self.predict_texts(
            texts,
            batch_size = batch_size,
        )

        return pd.concat(
            [
                df.reset_index(drop = True),
                pred_df.reset_index(drop = True),
            ],
            axis = 1,
        )

    def predict_one(
        self,
        title: str,
        description: str,
    ) -> Dict[str, Any]:
        df = pd.DataFrame([
            {
                "title": title,
                "description": description,
            }
        ])

        out = self.predict_df(
            df,
            title_col = "title",
            description_col = "description",
            batch_size = 1,
        )

        return out.iloc[0].to_dict()
