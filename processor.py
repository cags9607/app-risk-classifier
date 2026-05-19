import json
import logging
import os
import time
from typing import Any, Dict, List

from core import predict_records
from processor_config import (
    APP_CLASSIFIER_HF_REPO_ID,
    APP_CLASSIFIER_HF_TOKEN,
    APP_CLASSIFIER_INCLUDE_LABEL_IDS,
    APP_CLASSIFIER_INCLUDE_PROBABILITIES,
    APP_CLASSIFIER_LIST_MODELS,
    APP_CLASSIFIER_MAX_LENGTH,
    APP_CLASSIFIER_OVERWRITE_PREDICTION_ID,
    BATCH_SIZE,
)
from processor_utils import pop, push


logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s %(levelname)s %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


def _set_model_env():
    os.environ["APP_CLASSIFIER_HF_REPO_ID"] = APP_CLASSIFIER_HF_REPO_ID
    os.environ["APP_CLASSIFIER_LIST_MODELS"] = APP_CLASSIFIER_LIST_MODELS
    os.environ["APP_CLASSIFIER_MAX_LENGTH"] = str(APP_CLASSIFIER_MAX_LENGTH)
    os.environ["APP_CLASSIFIER_INCLUDE_PROBABILITIES"] = APP_CLASSIFIER_INCLUDE_PROBABILITIES
    os.environ["APP_CLASSIFIER_INCLUDE_LABEL_IDS"] = APP_CLASSIFIER_INCLUDE_LABEL_IDS
    os.environ["APP_CLASSIFIER_OVERWRITE_PREDICTION_ID"] = APP_CLASSIFIER_OVERWRITE_PREDICTION_ID

    if APP_CLASSIFIER_HF_TOKEN:
        os.environ["APP_CLASSIFIER_HF_TOKEN"] = APP_CLASSIFIER_HF_TOKEN


def _extract_payload(job: Dict[str, Any]) -> Dict[str, Any]:
    if "payload" in job and isinstance(job["payload"], dict):
        return job["payload"]

    return job


def _attach_result(job: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(job)

    if "result" in out and isinstance(out["result"], dict):
        out["result"].update(result)
    else:
        out["result"] = result

    return out


def process_jobs(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _set_model_env()

    payloads = [_extract_payload(job) for job in jobs]

    records = [
        {
            "prediction_id": payload.get("prediction_id"),
            "title": payload.get("title", ""),
            "description": payload.get("description", ""),
        }
        for payload in payloads
    ]

    predictions = predict_records(records)

    processed = [
        _attach_result(job, pred)
        for job, pred in zip(jobs, predictions)
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
