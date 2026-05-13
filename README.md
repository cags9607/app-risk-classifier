# app-classifier

Repo for the three app classifiers built from the notebook workflow:

1. prompt-label data with Llama 70B using `bundle_id + title + description`
2. fine-tune Qwen/Qwen2.5-3B-Instruct with QLoRA using `title + description` only
3. run prediction from HuggingFace-hosted adapter repos using `title + description` only
4. optionally run as a DeepSee-style queue processor

The repo intentionally does **not** use `bundle_id` during model inference. `bundle_id` is only used during LLM prompt-labeling.

## Install

```bash
pip install -e .
```

## Configs

Task configs live in `configs/`:

- `configs/incentivized.json`
- `configs/ai.json`
- `configs/pdu.json`

Each config has an `hf_model_id` placeholder. Replace it with the HuggingFace repo that stores the adapter/model assets.

## Predict CSV

```bash
python scripts/predict_csv.py \
  --model_id YOUR_ORG/YOUR_PDU_MODEL \
  --input_csv apps.csv \
  --output_csv predictions.csv \
  --title_col title \
  --description_col description \
  --batch_size 8
```

Output columns include:

- `pred_label_id`
- `pred_label`
- `pred_confidence`
- one `pred_prob_<label>` column per label

For binary tasks, it also adds:

- `pred_prob_positive`
- `pred_label_bin`

## Predict one app

```bash
python scripts/predict_one.py \
  --model_id YOUR_ORG/YOUR_INCENTIVIZED_MODEL \
  --title "Earn Money" \
  --description "Watch videos and earn PayPal cash"
```

## Queue processor

Root files mirror the review detector style:

- `core.py`
- `processor.py`
- `processor_config.py`
- `processor_utils.py`
- `dstack.yml`

Set environment variables:

```bash
export QUEUE_URL="https://deepsee-queue.herokuapp.com/exchange-batch"
export QUEUE_API_KEY="..."
export QUEUE_KEY="APP_CLASSIFIER_PLACEHOLDER"
export APP_CLASSIFIER_HF_REPO_ID="YOUR_ORG/YOUR_MODEL_REPO"
export APP_CLASSIFIER_TASK_CONFIG="configs/pdu.json"
python processor.py
```

Expected queue entries may be under `jobs`, `entries`, or `apps`. Each entry should contain at least `title` and `description`. `bundle_id` is passed through if present, but is not used by prediction.
