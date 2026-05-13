# File manifest

This archive is intentionally named `app-classifier-complete.zip` to avoid stale cached downloads.

Root-level worker/library files:

- `pyproject.toml`: installable Python package metadata.
- `core.py`: review-detector-style public classifier wrapper; loads HF model IDs through config.
- `processor.py`: queue processor loop.
- `processor_config.py`: environment-based queue/HF/runtime configuration with placeholders.
- `processor_utils.py`: queue pop/push helpers.
- `dstack.yml`: deployment placeholder.

Package files:

- `app_classifier/config.py`
- `app_classifier/inference.py`
- `app_classifier/labeling.py`
- `app_classifier/text.py`
- `app_classifier/training.py`

Scripts:

- `scripts/label_dataset.py`
- `scripts/train_qlora.py`
- `scripts/predict_csv.py`
- `scripts/predict_one.py`
- `scripts/zip_output_dir.py`

Task files:

- `configs/incentivized.json`
- `configs/ai.json`
- `configs/pdu.json`
- `prompts/incentivized.py`
- `prompts/ai.py`
- `prompts/pdu.py`
- `label_mappings/incentivized_label_mapping.json`
- `label_mappings/ai_label_mapping.json`
- `label_mappings/pdu_label_mapping.json`
