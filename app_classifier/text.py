import pandas as pd


def make_text(title, description) -> str:
    title = "" if title is None else str(title)
    description = "" if description is None else str(description)

    return f"title: {title}\ndescription: {description}"


def add_text_column(
    df: pd.DataFrame,
    title_col: str = "title",
    description_col: str = "description",
    text_col: str = "text",
) -> pd.DataFrame:
    d = df.copy()

    if title_col not in d.columns:
        raise ValueError(f"Missing title column: {title_col}")

    if description_col not in d.columns:
        raise ValueError(f"Missing description column: {description_col}")

    d[title_col] = d[title_col].fillna("").astype(str)
    d[description_col] = d[description_col].fillna("").astype(str)

    d[text_col] = (
        "title: " + d[title_col] +
        "\n" +
        "description: " + d[description_col]
    )

    return d
