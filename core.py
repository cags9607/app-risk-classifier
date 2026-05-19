import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from app_classifier.config import (
    env_bool,
    get_hf_token,
    normalize_cache_policy,
    normalize_hf_repo_id,
    parse_model_list,
)
from app_classifier.inference import MultiAppRiskClassifier


logger = logging.getLogger(__name__)

_CLASSIFIER_CACHE: Dict[Tuple[Any, ...], MultiAppRiskClassifier] = {}


def _get_cache_key(
    model_id: str,
    list_models: List[str],
    max_length: int,
    load_in_4bit: bool,
    cache_policy: str,
) -> Tuple[Any, ...]:
    return (
        model_id,
        tuple(list_models),
        max_length,
        load_in_4bit,
        cache_policy,
    )


def get_classifier(
    list_models: Optional[List[str]] = None,
    max_length: Optional[int] = None,
    load_in_4bit: Optional[bool] = None,
    cache_policy: Optional[str] = None,
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
        load_in_4bit = not env_bool("APP_CLASSIFIER_NO_4BIT", "0")

    cache_policy = normalize_cache_policy(cache_policy)

    cache_key = _get_cache_key(
        model_id = model_id,
        list_models = parsed_models,
        max_length = max_length,
        load_in_4bit = load_in_4bit,
        cache_policy = cache_policy,
    )

    if cache_key in _CLASSIFIER_CACHE:
        return _CLASSIFIER_CACHE[cache_key]

    token = get_hf_token()

    logger.info(
        "Creating MultiAppRiskClassifier: model_id=%s, models=%s, "
        "max_length=%s, load_in_4bit=%s, cache_policy=%s",
        model_id,
        parsed_models,
        max_length,
        load_in_4bit,
        cache_policy,
    )

    clf = MultiAppRiskClassifier.from_hf(
        model_id = model_id,
        list_models = parsed_models,
        token = token,
        max_length = max_length,
        load_in_4bit = load_in_4bit,
        cache_policy = cache_policy,
    )

    _CLASSIFIER_CACHE[cache_key] = clf

    return clf


def clear_classifier_cache():
    for clf in _CLASSIFIER_CACHE.values():
        try:
            clf.unload()
        except Exception:
            logger.exception("Failed to unload classifier during cache clear.")

    _CLASSIFIER_CACHE.clear()


def get_model_info() -> Dict[str, Any]:
    return {
        "n_cached_classifiers": len(_CLASSIFIER_CACHE),
        "cache_keys": [str(k) for k in _CLASSIFIER_CACHE.keys()],
        "classifiers": [
            clf.get_model_info()
            for clf in _CLASSIFIER_CACHE.values()
        ],
    }


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
        include_probabilities = env_bool("APP_CLASSIFIER_INCLUDE_PROBABILITIES", "0")

    if include_label_ids is None:
        include_label_ids = env_bool("APP_CLASSIFIER_INCLUDE_LABEL_IDS", "0")

    if overwrite_prediction_id is None:
        overwrite_prediction_id = env_bool("APP_CLASSIFIER_OVERWRITE_PREDICTION_ID", "0")

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
