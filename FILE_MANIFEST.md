# File manifest

This archive contains the simplified inference and queue-worker version of the app risk classifier repo.

## Root files

- `README.md`: usage and repo boundary.
- `FILE_MANIFEST.md`: this file.
- `pyproject.toml`: installable package metadata.
- `requirements.txt`: runtime dependencies.
- `env.example`: environment variable template.
- `dstack.yml`: queue-worker deployment template.
- `.gitignore`: ignored local/cache/data files.
- `core.py`: public prediction wrapper used by the worker.
- `processor.py`: queue processor loop.
- `processor_config.py`: environment-based queue and Hugging Face config.
- `processor_utils.py`: queue pop/push helpers.

## Package files

- `app_classifier/__init__.py`: exports `AppRiskClassifier`.
- `app_classifier/config.py`: Hugging Face repo/token helpers.
- `app_classifier/inference.py`: model loading and dataframe prediction.
- `app_classifier/text.py`: title/description text formatting.

## Scripts

- `scripts/predict_csv.py`: CSV batch inference script.

## Not included by design

- Training scripts.
- Prompt-labeling scripts.
- Local task configs.
- Local label mappings.
- Single-record `predict_one` script/function.

Label mappings now live in the Hugging Face repo beside each adapter subfolder.
