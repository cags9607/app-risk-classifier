import argparse

import pandas as pd

from app_classifier.config import parse_model_list, prompt_for_hf_repo_and_token
from app_classifier.inference import score_with_models


def main():
    parser = argparse.ArgumentParser(
        description = "Run app risk classifier predictions on a CSV file."
    )

    parser.add_argument(
        "--model_id",
        default = None,
        help = "Hugging Face repo ID or URL. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--list_models",
        nargs = "*",
        default = None,
        help = (
            "Models to run. Examples: "
            "--list_models pdu ai-powered incentivized "
            "or --list_models pdu,ai-powered,incentivized. "
            "Defaults to all models."
        ),
    )
    parser.add_argument(
        "--subfolder",
        default = None,
        help = (
            "Deprecated single-model shortcut. "
            "Use --list_models instead. Kept for compatibility."
        ),
    )
    parser.add_argument(
        "--hf_token",
        default = None,
        help = (
            "Hugging Face API key/token. If omitted, env vars are used; "
            "if none exist, you will be prompted."
        ),
    )
    parser.add_argument("--input_csv", required = True)
    parser.add_argument("--output_csv", required = True)
    parser.add_argument("--title_col", default = "title")
    parser.add_argument("--description_col", default = "description")
    parser.add_argument("--batch_size", type = int, default = 8)
    parser.add_argument("--max_length", type = int, default = 512)
    parser.add_argument("--no_4bit", action = "store_true")
    parser.add_argument(
        "--prediction_id_col",
        default = "prediction_id",
        help = "Name of the prediction ID column.",
    )
    parser.add_argument(
        "--overwrite_prediction_id",
        action = "store_true",
        help = "Always generate new prediction IDs, even if the input already has unique IDs.",
    )
    parser.add_argument(
        "--include_probabilities",
        action = "store_true",
        help = "Include one prefixed probability column per class for each model.",
    )
    parser.add_argument(
        "--include_label_ids",
        action = "store_true",
        help = "Include prefixed predicted label IDs for each model.",
    )
    parser.add_argument(
        "--ask_credentials",
        action = "store_true",
        help = (
            "Prompt for Hugging Face repo ID/URL and API key/token even if "
            "arguments or environment variables exist."
        ),
    )

    args = parser.parse_args()

    model_id, token = prompt_for_hf_repo_and_token(
        model_id = args.model_id,
        token = args.hf_token,
        force_prompt = args.ask_credentials,
    )

    if args.list_models is not None:
        list_models = parse_model_list(args.list_models)
    elif args.subfolder:
        list_models = parse_model_list([args.subfolder])
    else:
        list_models = parse_model_list(None)

    print("Running models:", ", ".join(list_models))

    df = pd.read_csv(args.input_csv)

    out = score_with_models(
        df = df,
        model_id = model_id,
        list_models = list_models,
        token = token,
        title_col = args.title_col,
        description_col = args.description_col,
        batch_size = args.batch_size,
        max_length = args.max_length,
        load_in_4bit = not args.no_4bit,
        prediction_id_col = args.prediction_id_col,
        overwrite_prediction_id = args.overwrite_prediction_id,
        include_probabilities = args.include_probabilities,
        include_label_ids = args.include_label_ids,
    )

    out.to_csv(args.output_csv, index = False)

    print(f"Saved predictions to: {args.output_csv}")

    preview_cols = [
        c for c in out.columns
        if (
            c == args.prediction_id_col
            or c in {args.title_col, args.description_col}
            or c.endswith("_pred_label")
            or c.endswith("_pred_confidence")
        )
    ]

    if preview_cols:
        print(out[preview_cols].head())


if __name__ == "__main__":
    main()
