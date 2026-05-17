import os
from getpass import getpass
from typing import Optional, Tuple
from urllib.parse import urlparse


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
) -> Tuple[str, Optional[str]]:
    repo_value = model_id

    while not repo_value:
        repo_value = input(
            "Hugging Face repo ID or URL "
            "(for example Trinotrotolueno/app-risk-adapters): "
        ).strip()

    repo_id = normalize_hf_repo_id(repo_value)

    hf_token = get_hf_token(token)

    if not hf_token:
        entered = getpass(
            "Hugging Face API key/token "
            "(press Enter if the repo is public): "
        ).strip()
        hf_token = entered or None

    return repo_id, hf_token
