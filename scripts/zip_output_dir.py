import argparse
import json
import pandas as pd

from app_classifier.config import load_task_config
from app_classifier.training import train_qlora_classifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--val_size", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()

    cfg = load_task_config(args.config)
    df = pd.read_csv(args.input_csv)

    result = train_qlora_classifier(
        df,
        sample_bucket_col=cfg.sample_bucket_col,
        title_col=cfg.title_col,
        description_col=cfg.description_col,
        label_col=cfg.label_col,
        mode=cfg.mode,
        model_name=cfg.model_name,
        output_dir=args.output_dir or cfg.output_dir,
        val_size=args.val_size,
        num_train_epochs=args.epochs,
        positive_labels=set(cfg.positive_labels or []),
        negative_labels=set(cfg.negative_labels or []),
    )

    print(json.dumps(result["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
