import argparse
import json
import os

from app_classifier.inference import AppRiskClassifier


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", required = True)
    parser.add_argument("--subfolder", default = None)
    parser.add_argument("--title", required = True)
    parser.add_argument("--description", required = True)
    parser.add_argument("--max_length", type = int, default = 512)
    parser.add_argument("--no_4bit", action = "store_true")

    args = parser.parse_args()

    token = (
        os.getenv("APP_CLASSIFIER_HF_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )

    clf = AppRiskClassifier.from_hf(
        model_id = args.model_id,
        subfolder = args.subfolder,
        token = token,
        max_length = args.max_length,
        load_in_4bit = not args.no_4bit,
    )

    out = clf.predict_one(
        title = args.title,
        description = args.description,
    )

    print(json.dumps(out, indent = 2, ensure_ascii = False))


if __name__ == "__main__":
    main()
