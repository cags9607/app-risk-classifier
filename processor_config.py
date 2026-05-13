import os


QUEUE_URL = os.getenv(
    "QUEUE_URL",
    "https://PLACEHOLDER_QUEUE_URL/exchange-batch",
)

QUEUE_API_KEY = os.getenv(
    "QUEUE_API_KEY",
    "PLACEHOLDER_QUEUE_API_KEY",
)

QUEUE_KEY = os.getenv(
    "QUEUE_KEY",
    "PLACEHOLDER_QUEUE_KEY",
)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

APP_CLASSIFIER_HF_REPO_ID = os.getenv(
    "APP_CLASSIFIER_HF_REPO_ID",
    "Trinotrotolueno/app-risk-adapters",
)

APP_CLASSIFIER_HF_SUBFOLDER = os.getenv(
    "APP_CLASSIFIER_HF_SUBFOLDER",
    "pdu",
)

APP_CLASSIFIER_HF_TOKEN = os.getenv(
    "APP_CLASSIFIER_HF_TOKEN",
    os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_HUB_TOKEN", "")),
)
