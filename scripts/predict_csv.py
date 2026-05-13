import argparse
import os

import pandas as pd

from app_classifier.inference import AppRiskClassifier


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", required = True)
    parser.add_argument("--subfolder", default = None)
    parser.add_argument("--input_csv", required = True)
    parser.add_argument("--output_csv", required = True)
    parser.add_argument("--title_col", default = "title")
    parser.add_argument("--description_col", default = "description")
    parser.add_argument("--batch_size", type = int, default = 8)
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

    df = pd.read_csv(args.input_csv)

    out = clf.predict_df(
        df,
        title_col = args.title_col,
        description_col = args.description_col,
        batch_size = args.batch_size,
    )

    out.to_csv(args.output_csv, index = False)

    print(f"Saved predictions to: {args.output_csv}")

    show_cols = [
        c for c in ["pred_label", "pred_confidence"]
        if c in out.columns
    ]

    if show_cols:
        print(out[show_cols].head())


if __name__ == "__main__":
    main()
