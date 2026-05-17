# app-risk-classifier

Inference and queue-worker repo for app risk classifiers trained elsewhere and hosted as Hugging Face PEFT adapter subfolders.

This repo intentionally does **not** include training scripts, prompt-labeling scripts, local task configs, or local label mappings. Label mappings are loaded from the Hugging Face model repo, next to each adapter subfolder.

## Final scope

This repo does three things:

1. Load a Hugging Face adapter model and its `label_mapping.json`.
2. Predict app risk labels from `title + description`.
3. Run the same prediction logic as a DeepSee-style queue worker.

Training, prompt design, and dataset labeling will be added later.

## Expected Hugging Face layout

```text
owner/app-risk-adapters/
├── pdu/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer files...
│   └── label_mapping.json
├── incentivized/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer files...
│   └── label_mapping.json
└── ai/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer files...
    └── label_mapping.json
```

`label_mapping.json` must contain both `label2id` and `id2label`.

## Install

```bash
pip install -e .
```

## Predict CSV

You can pass a Hugging Face repo ID:

```bash
python scripts/predict_csv.py \
  --model_id Trinotrotolueno/app-risk-adapters \
  --subfolder pdu \
  --input_csv apps.csv \
  --output_csv predictions.csv \
  --title_col title \
  --description_col description \
  --batch_size 8
```

You can also pass a Hugging Face URL:

```bash
python scripts/predict_csv.py \
  --model_id https://huggingface.co/Trinotrotolueno/app-risk-adapters \
  --subfolder pdu \
  --input_csv apps.csv \
  --output_csv predictions.csv
```

If `--model_id` is omitted, the script asks for the Hugging Face repo ID/URL. It also asks for a Hugging Face API key/token if none is provided through `--hf_token`, `APP_CLASSIFIER_HF_TOKEN`, `HF_TOKEN`, or `HUGGINGFACE_HUB_TOKEN`.

For private repos:

```bash
python scripts/predict_csv.py \
  --model_id Trinotrotolueno/app-risk-adapters \
  --subfolder pdu \
  --hf_token YOUR_HF_TOKEN \
  --input_csv apps.csv \
  --output_csv predictions.csv
```

Output columns include:

- `pred_label_id`
- `pred_label`
- `pred_confidence`
- one `pred_prob_<label>` column per class

## Queue processor

Set environment variables:

```bash
export QUEUE_URL="https://deepsee-queue.herokuapp.com/exchange-batch"
export QUEUE_API_KEY="..."
export QUEUE_KEY="APP_CLASSIFIER_PLACEHOLDER"
export APP_CLASSIFIER_HF_REPO_ID="Trinotrotolueno/app-risk-adapters"
export APP_CLASSIFIER_HF_SUBFOLDER="pdu"
export APP_CLASSIFIER_HF_TOKEN="..."
export BATCH_SIZE=8

python processor.py
```

Expected queue jobs can either be flat objects or objects with a `payload` dict. Each payload should contain at least:

```json
{
  "title": "App title",
  "description": "App description"
}
```

The worker attaches model outputs under the job's `result` field and pushes the processed jobs back to the queue.

## Repo structure

```text
app-risk-classifier-main/
├── README.md
├── FILE_MANIFEST.md
├── pyproject.toml
├── requirements.txt
├── env.example
├── dstack.yml
├── .gitignore
├── core.py
├── processor.py
├── processor_config.py
├── processor_utils.py
├── app_classifier/
│   ├── __init__.py
│   ├── config.py
│   ├── inference.py
│   └── text.py
└── scripts/
    └── predict_csv.py
```
