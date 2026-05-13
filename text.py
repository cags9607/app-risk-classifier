import os
import json
import re
import time
import random
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm.auto import tqdm
from openai import OpenAI


DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
INPUT_PRICE_PER_M = 0.0
OUTPUT_PRICE_PER_M = 0.0

DEFAULT_PROMPT = """
You are a strict classifier.

Return ONLY valid JSON with exactly the keys requested by the task prompt.
""".strip()


def estimate_cost_usd(
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    input_price_per_m: float = INPUT_PRICE_PER_M,
    output_price_per_m: float = OUTPUT_PRICE_PER_M,
) -> Optional[float]:
    if prompt_tokens is None or completion_tokens is None:
        return None
    return (
        (prompt_tokens / 1_000_000) * input_price_per_m +
        (completion_tokens / 1_000_000) * output_price_per_m
    )


def _extract_first_json_object(text: str) -> str:
    if text is None:
        raise ValueError("Model returned empty content.")

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {text[:500]}")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

    raise ValueError(f"Incomplete JSON object in response: {text[:500]}")


def _normalize_output_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _coerce_generic_field(key: str, value: Any):
    if value is None:
        return None

    key_lower = key.lower()

    if key_lower == "confidence":
        try:
            value = float(value)
            value = max(0.0, min(1.0, value))
            return value
        except Exception:
            return None

    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]

    if isinstance(value, dict):
        return value

    return value


def make_client(api_key: Optional[str] = None):
    api_key = api_key or os.getenv("DEEPINFRA_TOKEN")
    if not api_key:
        raise ValueError("Missing DeepInfra API key. Set DEEPINFRA_TOKEN or pass api_key = ...")

    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai",
    )


def label_description(
    description: Optional[str],
    title: Optional[str],
    bundle_id: Optional[str],
    output_cols: Dict[str, str],
    client=None,
    prompt: str = DEFAULT_PROMPT,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 120,
    use_json_mode: bool = True,
    return_raw: bool = False,
    input_price_per_m: float = INPUT_PRICE_PER_M,
    output_price_per_m: float = OUTPUT_PRICE_PER_M,
    strict_keys: bool = False,
) -> Dict[str, Any]:
    if client is None:
        client = make_client()

    description = "" if description is None else str(description)
    title = "" if title is None else str(title)
    bundle_id = "" if bundle_id is None else str(bundle_id)

    expected_keys = list(output_cols.keys())

    user_payload = {
        "title": title,
        "bundle_id": bundle_id,
        "description": description,
    }

    request_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "input": user_payload,
                        "required_output_keys": expected_keys,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if use_json_mode:
        request_kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**request_kwargs)
    raw_text = response.choices[0].message.content

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
    total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None

    estimated_cost = estimate_cost_usd(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        input_price_per_m=input_price_per_m,
        output_price_per_m=output_price_per_m,
    )

    parsed = json.loads(_extract_first_json_object(raw_text))

    out = {}
    for key in expected_keys:
        out[key] = _coerce_generic_field(key, parsed.get(key))

    if strict_keys:
        extra_keys = sorted(set(parsed.keys()) - set(expected_keys))
        missing_keys = sorted(set(expected_keys) - set(parsed.keys()))
        if extra_keys or missing_keys:
            raise ValueError(f"JSON keys mismatch. Missing: {missing_keys}; Extra: {extra_keys}")

    out["model"] = model
    out["prompt_tokens"] = prompt_tokens
    out["completion_tokens"] = completion_tokens
    out["total_tokens"] = total_tokens
    out["estimated_cost_usd"] = estimated_cost

    if return_raw:
        out["raw_response"] = raw_text

    return out


def _process_one_row(
    idx: int,
    row_dict: Dict[str, Any],
    description_col: str,
    title_col: str,
    bundle_id_col: str,
    output_cols: Dict[str, str],
    prompt: str,
    model: str,
    api_key: Optional[str],
    temperature: float,
    max_tokens: int,
    use_json_mode: bool,
    input_price_per_m: float,
    output_price_per_m: float,
    append_metadata: bool,
    return_raw: bool,
    max_retries: int,
    retry_sleep: float,
    strict_keys: bool,
):
    row_out = {dest_col: pd.NA for dest_col in output_cols.values()}

    if append_metadata:
        row_out["model"] = pd.NA
        row_out["prompt_tokens"] = pd.NA
        row_out["completion_tokens"] = pd.NA
        row_out["total_tokens"] = pd.NA
        row_out["estimated_cost_usd"] = pd.NA

    if return_raw:
        row_out["raw_response"] = pd.NA

    row_out["llm_success"] = False
    row_out["llm_attempts"] = 0
    row_out["llm_error"] = pd.NA
    row_out["llm_error_type"] = pd.NA

    client = make_client(api_key=api_key)

    for attempt in range(1, max_retries + 2):
        row_out["llm_attempts"] = attempt
        try:
            result = label_description(
                description=row_dict.get(description_col),
                title=row_dict.get(title_col),
                bundle_id=row_dict.get(bundle_id_col),
                output_cols=output_cols,
                client=client,
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                use_json_mode=use_json_mode,
                return_raw=return_raw,
                input_price_per_m=input_price_per_m,
                output_price_per_m=output_price_per_m,
                strict_keys=strict_keys,
            )

            for source_key, dest_col in output_cols.items():
                row_out[dest_col] = _normalize_output_value(result.get(source_key))

            if append_metadata:
                row_out["model"] = result.get("model")
                row_out["prompt_tokens"] = result.get("prompt_tokens")
                row_out["completion_tokens"] = result.get("completion_tokens")
                row_out["total_tokens"] = result.get("total_tokens")
                row_out["estimated_cost_usd"] = result.get("estimated_cost_usd")

            if return_raw:
                row_out["raw_response"] = result.get("raw_response")

            row_out["llm_success"] = True
            row_out["llm_error"] = pd.NA
            row_out["llm_error_type"] = pd.NA
            return idx, row_out

        except Exception as e:
            row_out["llm_error"] = str(e)
            row_out["llm_error_type"] = type(e).__name__
            if attempt < (max_retries + 1):
                sleep_s = retry_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
                time.sleep(sleep_s)
            else:
                return idx, row_out


def label_df(
    df: pd.DataFrame,
    description_col: str,
    title_col: str,
    bundle_id_col: str,
    output_cols: Dict[str, str],
    prompt: str = DEFAULT_PROMPT,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 120,
    use_json_mode: bool = True,
    input_price_per_m: float = INPUT_PRICE_PER_M,
    output_price_per_m: float = OUTPUT_PRICE_PER_M,
    append_metadata: bool = True,
    return_raw: bool = False,
    show_progress: bool = True,
    progress_desc: str = "Labeling rows",
    max_retries: int = 2,
    retry_sleep: float = 1.5,
    max_workers: int = 16,
    strict_keys: bool = False,
    save_every_n=None,
    save_path=None,
) -> pd.DataFrame:
    required_cols = [description_col, title_col, bundle_id_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df: {missing}")

    if not isinstance(output_cols, dict) or len(output_cols) == 0:
        raise ValueError("output_cols must be a non-empty dict like {'label': 'pred_label'}")

    records = df.to_dict(orient="records")
    results = [None] * len(records)

    running_cost = 0.0
    ok = 0
    fail = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                _process_one_row,
                idx,
                row_dict,
                description_col,
                title_col,
                bundle_id_col,
                output_cols,
                prompt,
                model,
                api_key,
                temperature,
                max_tokens,
                use_json_mode,
                input_price_per_m,
                output_price_per_m,
                append_metadata,
                return_raw,
                max_retries,
                retry_sleep,
                strict_keys,
            )
            for idx, row_dict in enumerate(records)
        ]

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc=progress_desc)

        for fut in iterator:
            idx, row_out = fut.result()
            results[idx] = row_out

            if bool(row_out["llm_success"]):
                ok += 1
                cost_i = row_out.get("estimated_cost_usd")
                if pd.notna(cost_i):
                    running_cost += float(cost_i)
            else:
                fail += 1

            n_done = ok + fail

            if save_every_n is not None and save_path is not None:
                if n_done % save_every_n == 0:
                    partial_results_df = pd.DataFrame(results)
                    partial_out = pd.concat(
                        [df.reset_index(drop=True), partial_results_df.reset_index(drop=True)],
                        axis=1,
                    )
                    partial_out.to_csv(save_path, index=False)

            if show_progress:
                try:
                    iterator.set_postfix(ok=ok, fail=fail, cost_usd=f"{running_cost:.6f}")
                except Exception:
                    pass

    results_df = pd.DataFrame(results)
    out = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

    if save_path is not None:
        out.to_csv(save_path, index=False)

    return out
