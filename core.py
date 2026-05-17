import os
from typing import Any, Dict, List, Optional

import pandas as pd

from app_classifier.config import get_hf_token, normalize_hf_repo_id
from app_classifier.inference import AppRiskClassifier


_CLASSIFIER: Optional[AppRiskClassifier] = None


def get_classifier() -> AppRiskClassifier:
    global _CLASSIFIER

    if _CLASSIFIER is not None:
        return _CLASSIFIER

    model_id = os.getenv("APP_CLASSIFIER_HF_REPO_ID")

    if not model_id:
        raise ValueError("Missing APP_CLASSIFIER_HF_REPO_ID.")

    model_id = normalize_hf_repo_id(model_id)

    subfolder = os.getenv("APP_CLASSIFIER_HF_SUBFOLDER") or None
    token = get_hf_token()

    _CLASSIFIER = AppRiskClassifier.from_hf(
        model_id = model_id,
        subfolder = subfolder,
        token = token,
    )

    return _CLASSIFIER


def predict_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clf = get_classifier()

    df = pd.DataFrame(records)

    if "title" not in df.columns:
        df["title"] = ""

    if "description" not in df.columns:
        df["description"] = ""

    out = clf.predict_df(
        df,
        title_col = "title",
        description_col = "description",
        batch_size = int(os.getenv("BATCH_SIZE", "8")),
    )

    return out.to_dict(orient = "records")
