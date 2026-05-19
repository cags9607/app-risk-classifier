import os
from copy import deepcopy
from getpass import getpass
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse


DEFAULT_MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "pdu": {
        "subfolder": "pdu",
        "prefix": "pdu",
    },
    "ai-powered": {
        "subfolder": "ai-powered",
        "prefix": "ai",
    },
    "incentivized": {
        "subfolder": "incentivized",
        "prefix": "incentivized",
    },
}

DEFAULT_LIST_MODELS = [
    "pdu",
    "ai-powered",
    "incentivized",
]


def get_hf_token(token: Optional[str] = None) -> Optional[str]:
    return (
        token
        or os.getenv("APP_CLASSIFIER_HF_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or None
    )


def normalize_hf_repo_id(repo_id_or_url: str) -> str:
    value = str(repo_id_or_url).strip()

    if not value:
        raise ValueError("Hugging Face repo ID/URL cannot be empty.")

    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        path_parts = [part for part in parsed.path.split("/") if part]

        if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
            raise ValueError(
                "Expected a Hugging Face URL like https://huggingface.co/org/repo."
            )

        if len(path_parts) < 2:
            raise ValueError(
                "Could not parse Hugging Face repo ID from URL. "
                "Expected https://huggingface.co/org/repo."
            )

        return "/".join(path_parts[:2])

    return value


def prompt_for_hf_repo_and_token(
    model_id: Optional[str] = None,
    token: Optional[str] = None,
    force_prompt: bool = False,
) -> Tuple[str, Optional[str]]:
    repo_value = None if force_prompt else model_id

    while not repo_value:
        repo_value = input(
            "Hugging Face repo ID or URL "
            "(for example Trinotrotolueno/app-risk-adapters): "
        ).strip()

    repo_id = normalize_hf_repo_id(repo_value)

    hf_token = None if force_prompt else get_hf_token(token)

    if not hf_token:
        entered = getpass(
            "Hugging Face API key/token "
            "(press Enter if the repo is public): "
        ).strip()
        hf_token = entered or None

    return repo_id, hf_token


def parse_model_list(
    value: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    if value is None:
        env_value = os.getenv("APP_CLASSIFIER_LIST_MODELS")

        if env_value:
            value = env_value
        else:
            return list(DEFAULT_LIST_MODELS)

    if isinstance(value, str):
        raw_parts = value.replace(";", ",").split(",")
        models = [part.strip() for part in raw_parts if part.strip()]
    else:
        models = []

        for item in value:
            if item is None:
                continue

            item = str(item).strip()

            if not item:
                continue

            if "," in item or ";" in item:
                models.extend(parse_model_list(item))
            else:
                models.append(item)

    if not models:
        return list(DEFAULT_LIST_MODELS)

    unknown = [
        model_name
        for model_name in models
        if model_name not in DEFAULT_MODEL_SPECS
    ]

    if unknown:
        raise ValueError(
            "Unknown model name(s): "
            f"{unknown}. Valid values are: {sorted(DEFAULT_MODEL_SPECS)}"
        )

    # Preserve order, remove duplicates.
    seen = set()
    deduped = []

    for model_name in models:
        if model_name not in seen:
            seen.add(model_name)
            deduped.append(model_name)

    return deduped


def get_model_specs(
    list_models: Optional[Union[str, List[str]]] = None,
) -> Dict[str, Dict[str, str]]:
    models = parse_model_list(list_models)

    return {
        model_name: deepcopy(DEFAULT_MODEL_SPECS[model_name])
        for model_name in models
    }
