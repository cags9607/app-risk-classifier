import gc
import json
import os
import uuid
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch

from huggingface_hub import hf_hub_download
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .config import (
    get_hf_token,
    get_model_specs,
    normalize_hf_repo_id,
    parse_model_list,
)
from .text import make_text


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


def _load_base_model_name_from_adapter(
    model_id: str,
    subfolder: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    peft_config = PeftConfig.from_pretrained(
        model_id,
        subfolder = subfolder,
        token = token,
    )

    base_model_name = getattr(peft_config, "base_model_name_or_path", None)

    if not base_model_name:
        raise ValueError(
            "Could not infer base_model_name_or_path from adapter_config.json."
        )

    return base_model_name


def _safe_prob_column_name(label: str) -> str:
    return (
        str(label)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def generate_prediction_ids(n: int) -> List[str]:
    return [uuid.uuid4().hex for _ in range(n)]


def ensure_prediction_id_column(
    df: pd.DataFrame,
    prediction_id_col: str = "prediction_id",
    overwrite: bool = False,
) -> pd.DataFrame:
    out = df.copy()

    needs_new_ids = (
        overwrite
        or prediction_id_col not in out.columns
        or out[prediction_id_col].isna().any()
        or out[prediction_id_col].astype(str).duplicated().any()
        or (out[prediction_id_col].astype(str).str.strip() == "").any()
    )

    if needs_new_ids:
        out[prediction_id_col] = generate_prediction_ids(len(out))

    return out


class AppRiskClassifier:
    def __init__(
        self,
        model_id: str,
        subfolder: Optional[str] = None,
        token: Optional[str] = None,
        base_model_name: Optional[str] = None,
        max_length: int = 512,
        device_map: str = "auto",
        load_in_4bit: bool = True,
    ):
        self.model_id = normalize_hf_repo_id(model_id)
        self.subfolder = subfolder
        self.token = get_hf_token(token)
        self.max_length = max_length

        self.base_model_name = (
            base_model_name
            or os.getenv("APP_CLASSIFIER_BASE_MODEL_NAME")
            or _load_base_model_name_from_adapter(
                model_id = self.model_id,
                subfolder = subfolder,
                token = self.token,
            )
        )

        self.label_mapping = _load_label_mapping(
            model_id = self.model_id,
            subfolder = subfolder,
            token = self.token,
        )

        self.label2id = self.label_mapping["label2id"]
        self.id2label = self.label_mapping["id2label"]
        self.num_labels = len(self.id2label)

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
            self.model_id,
            subfolder = subfolder,
            token = self.token,
            use_fast = True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading base model: {self.base_model_name}")
        print(f"Loading adapter repo: {self.model_id}")
        print(f"Loading adapter subfolder: {self.subfolder}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
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
            torch_dtype = compute_dtype,
            token = self.token,
        )

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.problem_type = "single_label_classification"

        self.model = PeftModel.from_pretrained(
            self.model,
            self.model_id,
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


class MultiAppRiskClassifier:
    def __init__(
        self,
        model_id: str,
        list_models: Optional[Union[str, List[str]]] = None,
        token: Optional[str] = None,
        max_length: int = 512,
        device_map: str = "auto",
        load_in_4bit: bool = True,
    ):
        self.model_id = normalize_hf_repo_id(model_id)
        self.list_models = parse_model_list(list_models)
        self.model_specs = get_model_specs(self.list_models)
        self.token = get_hf_token(token)
        self.max_length = max_length
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit

        self._classifiers: Dict[str, AppRiskClassifier] = {}

    @classmethod
    def from_hf(
        cls,
        model_id: str,
        list_models: Optional[Union[str, List[str]]] = None,
        token: Optional[str] = None,
        **kwargs,
    ):
        return cls(
            model_id = model_id,
            list_models = list_models,
            token = token,
            **kwargs,
        )

    def _get_classifier(self, model_name: str) -> AppRiskClassifier:
        if model_name in self._classifiers:
            return self._classifiers[model_name]

        spec = self.model_specs[model_name]

        clf = AppRiskClassifier.from_hf(
            model_id = self.model_id,
            subfolder = spec["subfolder"],
            token = self.token,
            max_length = self.max_length,
            device_map = self.device_map,
            load_in_4bit = self.load_in_4bit,
        )

        self._classifiers[model_name] = clf

        return clf

    def unload(self):
        self._classifiers.clear()
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict_df(
        self,
        df: pd.DataFrame,
        title_col: str = "title",
        description_col: str = "description",
        batch_size: int = 8,
        prediction_id_col: str = "prediction_id",
        overwrite_prediction_id: bool = False,
        include_probabilities: bool = False,
        include_label_ids: bool = False,
    ) -> pd.DataFrame:
        if title_col not in df.columns:
            raise ValueError(f"Missing title column: {title_col}")

        if description_col not in df.columns:
            raise ValueError(f"Missing description column: {description_col}")

        out = ensure_prediction_id_column(
            df = df,
            prediction_id_col = prediction_id_col,
            overwrite = overwrite_prediction_id,
        ).reset_index(drop = True)

        prediction_input = out[[title_col, description_col]].copy()

        for model_name in self.list_models:
            spec = self.model_specs[model_name]
            prefix = spec["prefix"]

            clf = self._get_classifier(model_name)

            pred_df = clf.predict_df(
                prediction_input,
                title_col = title_col,
                description_col = description_col,
                batch_size = batch_size,
            )

            model_cols = [
                col
                for col in pred_df.columns
                if col.startswith("pred_")
            ]

            if not include_probabilities:
                model_cols = [
                    col
                    for col in model_cols
                    if not col.startswith("pred_prob_")
                ]

            if not include_label_ids:
                model_cols = [
                    col
                    for col in model_cols
                    if col != "pred_label_id"
                ]

            rename_map = {
                col: f"{prefix}_{col}"
                for col in model_cols
            }

            model_out = (
                pred_df[model_cols]
                .rename(columns = rename_map)
                .reset_index(drop = True)
            )

            out = pd.concat(
                [
                    out.reset_index(drop = True),
                    model_out,
                ],
                axis = 1,
            )

        return out


def score_with_models(
    df: pd.DataFrame,
    model_id: str,
    list_models: Optional[Union[str, List[str]]] = None,
    token: Optional[str] = None,
    title_col: str = "title",
    description_col: str = "description",
    batch_size: int = 8,
    max_length: int = 500,
    load_in_4bit: bool = True,
    prediction_id_col: str = "prediction_id",
    overwrite_prediction_id: bool = False,
    include_probabilities: bool = False,
    include_label_ids: bool = False,
) -> pd.DataFrame:
    clf = MultiAppRiskClassifier.from_hf(
        model_id = model_id,
        list_models = list_models,
        token = token,
        max_length = max_length,
        load_in_4bit = load_in_4bit,
    )

    return clf.predict_df(
        df = df,
        title_col = title_col,
        description_col = description_col,
        batch_size = batch_size,
        prediction_id_col = prediction_id_col,
        overwrite_prediction_id = overwrite_prediction_id,
        include_probabilities = include_probabilities,
        include_label_ids = include_label_ids,
    )
