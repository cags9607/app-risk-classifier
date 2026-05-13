import argparse
import importlib
import pandas as pd

from app_classifier.config import load_task_config
from app_classifier.labeling import label_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompt_module", required=True, help="Example: prompts.incentivized")
    ap.add_argument("--prompt_name", required=True, help="Example: PROMPT_INCENTIVIZED")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--max_workers", type=int, default=16)
    ap.add_argument("--save_every_n", type=int, default=None)
    args = ap.parse_args()

    cfg = load_task_config(args.config)
    prompt = getattr(importlib.import_module(args.prompt_module), args.prompt_name)
    df = pd.read_csv(args.input_csv)

    out = label_df(
        df=df,
        description_col=cfg.description_col,
        title_col=cfg.title_col,
        bundle_id_col=cfg.bundle_id_col,
        output_cols=cfg.output_cols,
        prompt=prompt,
        max_workers=args.max_workers,
        save_every_n=args.save_every_n,
        save_path=args.output_csv,
    )
    out.to_csv(args.output_csv, index=False)
    print(f"Saved labeled data to: {args.output_csv}")


if __name__ == "__main__":
    main()
