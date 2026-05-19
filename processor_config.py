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

APP_CLASSIFIER_CACHE_POLICY = os.getenv(
    "APP_CLASSIFIER_CACHE_POLICY",
    "keep",
)

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

APP_CLASSIFIER_NO_4BIT = os.getenv(
    "APP_CLASSIFIER_NO_4BIT",
    "0",
)

# Queue identity behavior.
# prediction_id is generated internally and removed before pushing results.
# bundle_id is the external identifier returned downstream.
APP_CLASSIFIER_REQUIRE_BUNDLE_ID = os.getenv(
    "APP_CLASSIFIER_REQUIRE_BUNDLE_ID",
    "1",
)

APP_CLASSIFIER_DUPLICATE_BUNDLE_ID_POLICY = os.getenv(
    "APP_CLASSIFIER_DUPLICATE_BUNDLE_ID_POLICY",
    "error",
)
