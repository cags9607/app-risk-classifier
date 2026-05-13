import argparse
import pandas as pd

from app_classifier.inference import AppClassifier, InferConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--title_col", default="title")
    ap.add_argument("--description_col", default="description")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--hf_token", default=None)
    ap.add_argument("--label_mapping_path", default=None)
    ap.add_argument("--base_model_name", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--no_4bit", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    clf = AppClassifier(InferConfig(
        model_id=args.model_id,
        revision=args.revision,
        hf_token=args.hf_token,
        batch_size=args.batch_size,
        max_length=args.max_length,
        label_mapping_path=args.label_mapping_path,
        base_model_name=args.base_model_name,
        load_in_4bit=not args.no_4bit,
    )).load_model()

    out = clf.predict_df(df, title_col=args.title_col, description_col=args.description_col)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
