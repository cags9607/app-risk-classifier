from typing import Any
import pandas as pd


def coerce_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def merge_title_description(title: Any, description: Any) -> str:
    return "title: " + coerce_text(title) + "\n" + "description: " + coerce_text(description)


def add_model_text_column(
    df: pd.DataFrame,
    *,
    title_col: str = "title",
    description_col: str = "description",
    text_col: str = "text",
) -> pd.DataFrame:
    d = df.copy()
    d[title_col] = d[title_col].fillna("").astype(str)
    d[description_col] = d[description_col].fillna("").astype(str)
    d[text_col] = "title: " + d[title_col] + "\n" + "description: " + d[description_col]
    return d
