import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app_classifier.config import (
    get_hf_token,
    normalize_hf_repo_id,
    parse_model_list,
)
from app_classifier.inference import MultiAppRiskClassifier


_CLASSIFIER_CACHE: Dict[Tuple[Any, ...], MultiAppRiskClassifier] = {}


def _get_cache_key(
    model_id: str,
    list_models: List[str],
    max_length: int,
    load_in_4bit: bool,
) -> Tuple[Any, ...]:
    return (
        model_id,
        tuple(list_models),
        max_length,
        load_in_4bit,
    )


def get_classifier(
    list_models: Optional[List[str]] = None,
    max_length: Optional[int] = None,
    load_in_4bit: Optional[bool] = None,
) -> MultiAppRiskClassifier:
    model_id = os.getenv("APP_CLASSIFIER_HF_REPO_ID")

    if not model_id:
        raise ValueError("Missing APP_CLASSIFIER_HF_REPO_ID.")

    model_id = normalize_hf_repo_id(model_id)

    parsed_models = parse_model_list(
        list_models
        or os.getenv("APP_CLASSIFIER_LIST_MODELS")
    )

    max_length = max_length or int(os.getenv("APP_CLASSIFIER_MAX_LENGTH", "512"))

    if load_in_4bit is None:
        load_in_4bit = os.getenv("APP_CLASSIFIER_NO_4BIT", "0") not in {"1", "true", "TRUE"}

    cache_key = _get_cache_key(
        model_id = model_id,
        list_models = parsed_models,
        max_length = max_length,
        load_in_4bit = load_in_4bit,
    )

    if cache_key in _CLASSIFIER_CACHE:
        return _CLASSIFIER_CACHE[cache_key]

    token = get_hf_token()

    clf = MultiAppRiskClassifier.from_hf(
        model_id = model_id,
        list_models = parsed_models,
        token = token,
        max_length = max_length,
        load_in_4bit = load_in_4bit,
    )

    _CLASSIFIER_CACHE[cache_key] = clf

    return clf


def predict_records(
    records: List[Dict[str, Any]],
    list_models: Optional[List[str]] = None,
    include_probabilities: Optional[bool] = None,
    include_label_ids: Optional[bool] = None,
    overwrite_prediction_id: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    if not records:
        return []

    clf = get_classifier(list_models = list_models)

    df = pd.DataFrame(records)

    if "title" not in df.columns:
        df["title"] = ""

    if "description" not in df.columns:
        df["description"] = ""

    batch_size = int(os.getenv("BATCH_SIZE", "8"))

    if include_probabilities is None:
        include_probabilities = os.getenv("APP_CLASSIFIER_INCLUDE_PROBABILITIES", "0") in {
            "1",
            "true",
            "TRUE",
        }

    if include_label_ids is None:
        include_label_ids = os.getenv("APP_CLASSIFIER_INCLUDE_LABEL_IDS", "0") in {
            "1",
            "true",
            "TRUE",
        }

    if overwrite_prediction_id is None:
        overwrite_prediction_id = os.getenv("APP_CLASSIFIER_OVERWRITE_PREDICTION_ID", "0") in {
            "1",
            "true",
            "TRUE",
        }

    out = clf.predict_df(
        df,
        title_col = "title",
        description_col = "description",
        batch_size = batch_size,
        prediction_id_col = "prediction_id",
        overwrite_prediction_id = overwrite_prediction_id,
        include_probabilities = include_probabilities,
        include_label_ids = include_label_ids,
    )

    return out.to_dict(orient = "records")
