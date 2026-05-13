import time
import logging
from typing import Any, Dict, List, Optional

from processor_utils import pop, push
from processor_config import (
    BATCH_SIZE,
    EMPTY_QUEUE_SLEEP_SECONDS,
    APP_CLASSIFIER_HF_REPO_ID,
    APP_CLASSIFIER_REVISION,
    APP_CLASSIFIER_HF_TOKEN,
    APP_CLASSIFIER_TASK_CONFIG,
    APP_CLASSIFIER_LABEL_MAPPING_PATH,
    APP_CLASSIFIER_DEVICE,
    APP_CLASSIFIER_MAX_LENGTH,
    APP_CLASSIFIER_MODEL_BATCH_SIZE,
    APP_CLASSIFIER_LOAD_IN_4BIT,
    APP_CLASSIFIER_PREFER_BF16,
)

from app_classifier.config import load_task_config
from core import AppRiskClassifier, AppInferConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

classifier = None
task_config = None


def _init_classifier_if_needed():
    global classifier, task_config

    if classifier is not None:
        return

    task_config = load_task_config(APP_CLASSIFIER_TASK_CONFIG)

    repo_id = APP_CLASSIFIER_HF_REPO_ID
    if repo_id == "YOUR_ORG/YOUR_MODEL_REPO" and task_config.hf_model_id:
        repo_id = task_config.hf_model_id

    label_mapping_path = APP_CLASSIFIER_LABEL_MAPPING_PATH or task_config.label_mapping_path

    positive_label = None
    if task_config.positive_labels:
        positive_label = task_config.positive_labels[0]

    logger.info(f"Initializing app classifier from HF repo: {repo_id}")

    cfg = AppInferConfig(
        repo_id=repo_id,
        revision=APP_CLASSIFIER_REVISION,
        hf_token=APP_CLASSIFIER_HF_TOKEN,
        device=APP_CLASSIFIER_DEVICE,
        max_length=APP_CLASSIFIER_MAX_LENGTH,
        batch_size=APP_CLASSIFIER_MODEL_BATCH_SIZE,
        load_in_4bit=APP_CLASSIFIER_LOAD_IN_4BIT,
        prefer_bf16=APP_CLASSIFIER_PREFER_BF16,
        base_model_name=task_config.model_name,
        label_mapping_path=label_mapping_path,
        mode=task_config.mode,
        positive_label=positive_label,
    )

    classifier = AppRiskClassifier(cfg=cfg)
    classifier.load_models()
    logger.info("App classifier initialized successfully.")


def _extract_entries(job_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries = (
        job_data.get("jobs")
        or job_data.get("entries")
        or job_data.get("apps")
    )

    if entries is None:
        entries = [job_data]

    if not isinstance(entries, list):
        raise ValueError("Expected job data to contain a list under jobs, entries, or apps.")

    return entries


def _get_prediction_id(entry: Dict[str, Any], fallback: str) -> str:
    return str(
        entry.get("prediction_id")
        or entry.get("app_id")
        or entry.get("bundle_id")
        or entry.get("id")
        or fallback
    )


def _empty_prediction(
    *,
    prediction_id: str,
    status: str = "empty_or_failed",
    error: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "status": status,
        "prediction_id": prediction_id,
        "pred_label_id": None,
        "pred_label": None,
        "pred_confidence": None,
        "error": error,
    }


def _build_results(
    *,
    entries: List[Dict[str, Any]],
    prediction_ids: List[str],
    preds: List[Dict[str, Any]],
    job_id: str,
) -> List[Dict[str, Any]]:
    preds_by_id = {}
    for pred in preds:
        pid = str(pred.get("prediction_id") or "")
        if pid and pid not in preds_by_id:
            preds_by_id[pid] = pred

    results = []
    for i, entry in enumerate(entries):
        prediction_id = prediction_ids[i]
        pred = preds_by_id.get(prediction_id)
        if pred is None:
            pred = _empty_prediction(prediction_id=prediction_id, error="missing_prediction")

        results.append({
            "job_id": job_id,
            "prediction_id": prediction_id,
            "bundle_id": entry.get("bundle_id"),
            "title": entry.get("title"),
            "description": entry.get("description"),
            **{k: v for k, v in pred.items() if k != "prediction_id"},
        })

    return results


def _push_sub_batch(
    *,
    results: List[Dict[str, Any]],
    job_id: str,
    job_token: Optional[str],
    ack: bool,
):
    jobs_payload = [{"id": job_id, "token": job_token}] if ack else []

    processed_jobs = [
        {
            "jobs": jobs_payload,
            "filename": f"app_classifier_results_{job_id}_{int(time.time())}.json",
            "results": results,
        }
    ]

    push(processed_jobs)


def process_batch():
    _init_classifier_if_needed()

    jobs = pop(batch_size=1)

    if len(jobs) == 0:
        logger.info("No jobs received from queue. Sleeping.")
        time.sleep(EMPTY_QUEUE_SLEEP_SECONDS)
        return

    job = jobs[0]
    job_id = str(job["id"])
    job_token = job.get("token")
    job_data = job.get("data") or {}

    entries = _extract_entries(job_data)
    total = len(entries)

    logger.info(f"Processing app-classifier job {job_id}: {total} entries")

    if total == 0:
        logger.warning(f"Job {job_id} has 0 entries. Acking immediately.")
        _push_sub_batch(results=[], job_id=job_id, job_token=job_token, ack=True)
        return

    total_pushed = 0

    for chunk_start in range(0, total, BATCH_SIZE):
        chunk = entries[chunk_start:chunk_start + BATCH_SIZE]

        prediction_ids = [
            _get_prediction_id(entry, fallback=f"{job_id}_{chunk_start + i}")
            for i, entry in enumerate(chunk)
        ]

        try:
            preds = classifier.classify_apps_batch(
                entries=chunk,
                prediction_ids=prediction_ids,
            )
        except Exception as e:
            logger.exception(f"Prediction failed for job {job_id} chunk {chunk_start}")
            preds = [
                _empty_prediction(prediction_id=pid, error=str(e))
                for pid in prediction_ids
            ]

        results = _build_results(
            entries=chunk,
            prediction_ids=prediction_ids,
            preds=preds,
            job_id=job_id,
        )

        is_last = chunk_start + BATCH_SIZE >= total
        _push_sub_batch(
            results=results,
            job_id=job_id,
            job_token=job_token,
            ack=is_last,
        )

        total_pushed += len(results)
        logger.info(f"Pushed {len(results)} results for job {job_id}. total_pushed={total_pushed}/{total}")


def main():
    while True:
        try:
            process_batch()
        except KeyboardInterrupt:
            raise
        except Exception:
            logger.exception("Processor loop failed. Sleeping before retry.")
            time.sleep(EMPTY_QUEUE_SLEEP_SECONDS)


if __name__ == "__main__":
    main()
