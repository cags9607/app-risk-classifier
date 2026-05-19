import json
import logging
import os
import time
import uuid
from collections import Counter
from typing import Any, Dict, List

from core import predict_records
from processor_config import (
    APP_CLASSIFIER_CACHE_POLICY,
    APP_CLASSIFIER_DUPLICATE_BUNDLE_ID_POLICY,
    APP_CLASSIFIER_HF_REPO_ID,
    APP_CLASSIFIER_HF_TOKEN,
    APP_CLASSIFIER_INCLUDE_LABEL_IDS,
    APP_CLASSIFIER_INCLUDE_PROBABILITIES,
    APP_CLASSIFIER_LIST_MODELS,
    APP_CLASSIFIER_MAX_LENGTH,
    APP_CLASSIFIER_NO_4BIT,
    APP_CLASSIFIER_OVERWRITE_PREDICTION_ID,
    APP_CLASSIFIER_REQUIRE_BUNDLE_ID,
    BATCH_SIZE,
)
from processor_utils import pop, push


logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s %(levelname)s %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


def _env_bool(value: str) -> bool:
    return str(value).strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
        "y",
        "Y",
    }


def _set_model_env():
    os.environ["APP_CLASSIFIER_HF_REPO_ID"] = APP_CLASSIFIER_HF_REPO_ID
    os.environ["APP_CLASSIFIER_LIST_MODELS"] = APP_CLASSIFIER_LIST_MODELS
    os.environ["APP_CLASSIFIER_MAX_LENGTH"] = str(APP_CLASSIFIER_MAX_LENGTH)
    os.environ["APP_CLASSIFIER_CACHE_POLICY"] = APP_CLASSIFIER_CACHE_POLICY
    os.environ["APP_CLASSIFIER_INCLUDE_PROBABILITIES"] = APP_CLASSIFIER_INCLUDE_PROBABILITIES
    os.environ["APP_CLASSIFIER_INCLUDE_LABEL_IDS"] = APP_CLASSIFIER_INCLUDE_LABEL_IDS
    os.environ["APP_CLASSIFIER_OVERWRITE_PREDICTION_ID"] = APP_CLASSIFIER_OVERWRITE_PREDICTION_ID
    os.environ["APP_CLASSIFIER_NO_4BIT"] = APP_CLASSIFIER_NO_4BIT

    if APP_CLASSIFIER_HF_TOKEN:
        os.environ["APP_CLASSIFIER_HF_TOKEN"] = APP_CLASSIFIER_HF_TOKEN


def _extract_payload(job: Dict[str, Any]) -> Dict[str, Any]:
    if "payload" in job and isinstance(job["payload"], dict):
        return job["payload"]

    return job


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def _validate_duplicate_bundle_policy():
    valid_policies = {"error"}

    if APP_CLASSIFIER_DUPLICATE_BUNDLE_ID_POLICY not in valid_policies:
        raise ValueError(
            "Invalid APP_CLASSIFIER_DUPLICATE_BUNDLE_ID_POLICY="
            f"{APP_CLASSIFIER_DUPLICATE_BUNDLE_ID_POLICY!r}. "
            f"Valid values are: {sorted(valid_policies)}"
        )


def _validate_bundle_ids(payloads: List[Dict[str, Any]]):
    _validate_duplicate_bundle_policy()

    require_bundle_id = _env_bool(APP_CLASSIFIER_REQUIRE_BUNDLE_ID)

    bundle_ids = [
        _coerce_text(payload.get("bundle_id"))
        for payload in payloads
    ]

    if require_bundle_id:
        missing_positions = [
            i
            for i, bundle_id in enumerate(bundle_ids)
            if not bundle_id
        ]

        if missing_positions:
            raise ValueError(
                "Missing bundle_id in queue payload(s) at batch positions: "
                f"{missing_positions[:20]}"
            )

    non_empty_bundle_ids = [
        bundle_id
        for bundle_id in bundle_ids
        if bundle_id
    ]

    counts = Counter(non_empty_bundle_ids)

    duplicates = {
        bundle_id: n
        for bundle_id, n in counts.items()
        if n > 1
    }

    if duplicates and APP_CLASSIFIER_DUPLICATE_BUNDLE_ID_POLICY == "error":
        duplicate_preview = dict(list(duplicates.items())[:20])

        raise ValueError(
            "Duplicate bundle_id values found in the same queue batch. "
            "This would make output alignment ambiguous. "
            f"Duplicate preview: {duplicate_preview}"
        )


def _build_records(payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            # Internal only. Used by inference for row alignment.
            # It is removed before the result is pushed.
            "prediction_id": uuid.uuid4().hex,

            # External identifier. This is returned downstream.
            "bundle_id": _coerce_text(payload.get("bundle_id")),

            "title": _coerce_text(payload.get("title")),
            "description": _coerce_text(payload.get("description")),
        }
        for payload in payloads
    ]


def _clean_prediction_result(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Queue output contract:
      - bundle_id
      - title
      - description
      - *_pred_label
      - *_pred_confidence

    Optional, if enabled by env:
      - *_pred_prob_*
      - *_pred_label_id

    Internal prediction_id is never returned.
    """
    include_probabilities = _env_bool(APP_CLASSIFIER_INCLUDE_PROBABILITIES)
    include_label_ids = _env_bool(APP_CLASSIFIER_INCLUDE_LABEL_IDS)

    cleaned = {}

    for col in ["bundle_id", "title", "description"]:
        cleaned[col] = prediction.get(col)

    for key, value in prediction.items():
        if key == "prediction_id":
            continue

        if key in {"bundle_id", "title", "description"}:
            continue

        if key.endswith("_pred_label"):
            cleaned[key] = value
            continue

        if key.endswith("_pred_confidence"):
            cleaned[key] = value
            continue

        if include_label_ids and key.endswith("_pred_label_id"):
            cleaned[key] = value
            continue

        if include_probabilities and "_pred_prob_" in key:
            cleaned[key] = value
            continue

    return cleaned


def _attach_result(job: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(job)

    if "result" in out and isinstance(out["result"], dict):
        out["result"].update(result)
    else:
        out["result"] = result

    return out


def process_jobs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _set_model_env()

    if not jobs:
        return []

    payloads = [_extract_payload(job) for job in jobs]

    _validate_bundle_ids(payloads)

    records = _build_records(payloads)

    predictions = predict_records(records)

    cleaned_predictions = [
        _clean_prediction_result(pred)
        for pred in predictions
    ]

    processed = [
        _attach_result(job, pred)
        for job, pred in zip(jobs, cleaned_predictions)
    ]

    return processed


def run_once() -> int:
    _set_model_env()

    jobs = pop(batch_size = BATCH_SIZE)

    if not jobs:
        logger.info("No jobs received.")
        return 0

    logger.info("Pulled %s jobs.", len(jobs))

    processed_jobs = process_jobs(jobs)

    response = push(processed_jobs)

    logger.info("Pushed %s processed jobs.", len(processed_jobs))
    logger.info("Queue response: %s", json.dumps(response, ensure_ascii = False)[:1000])

    return len(processed_jobs)


def main():
    _set_model_env()

    logger.info(
        "Starting app classifier processor: models=%s, cache_policy=%s, batch_size=%s, "
        "require_bundle_id=%s, duplicate_bundle_id_policy=%s",
        APP_CLASSIFIER_LIST_MODELS,
        APP_CLASSIFIER_CACHE_POLICY,
        BATCH_SIZE,
        APP_CLASSIFIER_REQUIRE_BUNDLE_ID,
        APP_CLASSIFIER_DUPLICATE_BUNDLE_ID_POLICY,
    )

    while True:
        try:
            n = run_once()

            if n == 0:
                time.sleep(5)

        except KeyboardInterrupt:
            logger.info("Stopping processor.")
            break

        except Exception as e:
            logger.exception("Processor error: %s", e)
            time.sleep(10)


if __name__ == "__main__":
    main()
