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

APP_CLASSIFIER_LIST_MODELS = os.getenv(
    "APP_CLASSIFIER_LIST_MODELS",
    "pdu,ai-powered,incentivized",
)

APP_CLASSIFIER_HF_TOKEN = os.getenv(
    "APP_CLASSIFIER_HF_TOKEN",
    os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_HUB_TOKEN", "")),
)

APP_CLASSIFIER_MAX_LENGTH = int(os.getenv("APP_CLASSIFIER_MAX_LENGTH", "512"))

APP_CLASSIFIER_INCLUDE_PROBABILITIES = os.getenv(
    "APP_CLASSIFIER_INCLUDE_PROBABILITIES",
    "0",
)

APP_CLASSIFIER_INCLUDE_LABEL_IDS = os.getenv(
    "APP_CLASSIFIER_INCLUDE_LABEL_IDS",
    "0",
)

APP_CLASSIFIER_OVERWRITE_PREDICTION_ID = os.getenv(
    "APP_CLASSIFIER_OVERWRITE_PREDICTION_ID",
    "0",
)
