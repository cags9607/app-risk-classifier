import logging
from typing import Any, Dict, List

import requests

from processor_config import QUEUE_API_KEY, QUEUE_KEY, QUEUE_URL


logger = logging.getLogger(__name__)


def pop(batch_size: int = 1) -> List[Dict[str, Any]]:
    payload = {
        "key": QUEUE_KEY,
        "get": batch_size,
    }

    headers = {
        "x-api-key": QUEUE_API_KEY,
    }

    resp = requests.post(
        QUEUE_URL,
        json = payload,
        headers = headers,
        timeout = 60,
    )

    resp.raise_for_status()

    data = resp.json()

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["jobs", "data", "items"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError(f"Unexpected queue pop response shape: {type(data)}")


def push(processed_jobs: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = {
        "key": QUEUE_KEY,
        "put": processed_jobs,
    }

    headers = {
        "x-api-key": QUEUE_API_KEY,
    }

    resp = requests.post(
        QUEUE_URL,
        json = payload,
        headers = headers,
        timeout = 60,
    )

    resp.raise_for_status()

    return resp.json()
