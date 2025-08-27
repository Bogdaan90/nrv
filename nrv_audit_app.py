import os
import re
import io
import json
from typing import Dict, List, Tuple, Optional, Any, Iterable, Union

import pandas as pd
import streamlit as st

# OpenAI SDK (Responses API)
try:
    from openai import OpenAI

    _openai_available = True
except Exception:
    _openai_available = False

APP_TITLE = "NRV Audit — Product Label Compliance (EU/UK)"
DEFAULT_MODEL = "gpt-5-mini"  # choose any Responses-capable text model
AI_ENABLED_DEFAULT = True
AI_MAX_ROWS_DEFAULT = 500

# --- Canonical EU/UK NRV nutrients for vitamins & minerals --------------------
VITAMIN_MINERAL_NRV_TERMS = {
    # Vitamins
    "vitamin a",
    "retinol",
    "beta carotene",
    "vitamin d",
    "cholecalciferol",
    "ergocalciferol",
    "vitamin e",
    "tocopherol",
    "alpha tocopherol",
    "vitamin k",
    "phylloquinone",
    "menaquinone",
    "vitamin c",
    "ascorbic acid",
    "thiamin",
    "thiamine",
    "vitamin b1",
    "riboflavin",
    "vitamin b2",
    "niacin",
    "nicotinamide",
    "nicotinic acid",
    "vitamin b6",
    "pyridoxine",
    "folate",
    "folic acid",
    "vitamin b12",
    "cobalamin",
    "biotin",
    "pantothenic acid",
    "pantothenate",
    # Minerals
    "potassium",
    "k",
    "chloride",
    "cl",
    "calcium",
    "ca",
    "phosphorus",
    "phosphate",
    "p",
    "magnesium",
    "mg",
    "iron",
    "fe",
    "zinc",
    "zn",
    "copper",
    "cu",
    "manganese",
    "mn",
    "fluoride",
    "f",
    "selenium",
    "se",
    "chromium",
    "cr",
    "molybdenum",
    "mo",
    "iodine",
    "i",
}

# Non-food heuristics; if name looks like non-food, NRV isn't required
NON_FOOD_TOKENS = {
    "beauty",
    "skin",
    "hair",
    "cosmetic",
    "body care",
    "makeup",
    "skincare",
    "device",
    "test kit",
    "accessory",
    "equipment",
    "appliance",
    "aromatherapy",
    "diffuser",
    "fragrance",
    "perfume",
    "topical",
    "balm",
    "ointment",
    "cream",
    "gel",
    "serum",
    "soap",
}

# Regex for free-text NRV detection (fallback)
NRV_REGEX = re.compile(r"(\d+(?:\.\d+)?)\s*%?\s*NRV", flags=re.IGNORECASE)

Number = Union[int, float]

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------


def normalise_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def normalize_label(s: Any) -> str:
    if s is None:
        return ""
    t = str(s).lower()
    t = t.replace("_", " ").replace("-", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def strip_nrv_suffixes(s: str) -> str:
    """
    Remove common NRV/RI/RDA suffix noise from keys like 'vitamin_c_nrv', 'vitamin c nrv percent'.
    """
    t = normalize_label(s)
    t = re.sub(r"\b(nrv|ri|rda)( percent| %| pct)?\b.*$", "", t).strip()
    return t


def looks_non_food(name_or_category: str) -> bool:
    blob = normalize_label(name_or_category or "")
    return any(tok in blob for tok in NON_FOOD_TOKENS)


def mentions_vitmin(text: str) -> bool:
    if not text:
        return False
    t = normalize_label(text)
    return any(term in t for term in VITAMIN_MINERAL_NRV_TERMS)


def stringify_maybe_json(val: Any) -> str:
    if isinstance(val, (dict, list)):
        try:
            return json.dumps(val, ensure_ascii=False).lower()
        except Exception:
            return str(val).lower()
    return normalise_text(val)


def approx_tokens_from_text(text: str) -> int:
    """Rough heuristic: ~4 chars per token (conservative)."""
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def make_json_safe(obj: Any) -> Any:
    """
    Convert pandas/NumPy/complex objects into JSON-serializable primitives.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, dict):
        safe = {}
        for k, v in obj.items():
            try:
                key = k if isinstance(k, (str, int, float, bool)) else str(k)
            except Exception:
                key = str(k)
            safe[key] = make_json_safe(v)
        return safe
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    return str(obj)


# ------------------------------------------------------------------------------
# JSON parsing for nutritional_info
# ------------------------------------------------------------------------------

NRV_KEY_HINTS = {
    "nrv",
    "nrv%",
    "%nrv",
    "nrv percent",
    "nrv_percent",
    "percent nrv",
    "percent_nrv",
    "daily",
    "ri",
    "reference intake",
    "rda",
}


def _maybe_number(x: Any) -> Optional[Number]:
    if isinstance(x, (int, float)) and not pd.isna(x):
        return float(x)
    if isinstance(x, str):
        m = re.search(r"(\d+(?:\.\d+)?)", x)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None


def _is_vitmin_name(n: str) -> bool:
    return mentions_vitmin(strip_nrv_suffixes(n))


def _scan_for_percents(obj: Any) -> List[float]:
    """Extract plausible % values from nested objects (dict/list/str/num)."""
    found: List[float] = []

    def add_if_number(v: Any):
        num = _maybe_number(v)
        if num is not None:
            found.append(num)

    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = normalize_label(k)
            if any(h in lk for h in NRV_KEY_HINTS) or "%" in lk:
                add_if_number(v)
            if isinstance(v, (dict, list, str, int, float)):
                found.extend(_scan_for_percents(v))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_scan_for_percents(item))
    elif isinstance(obj, (int, float, str)):
        if isinstance(obj, str):
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%?", obj):
                try:
                    found.append(float(m.group(1)))
                except Exception:
                    pass
        else:
            add_if_number(obj)
    return found


def _iter_nutrients(obj: Any) -> Iterable[Tuple[str, Any]]:
    """
    Yield (nutrient_name, value_block) from varied shapes:
      - { "Vitamin C": {...}, "Magnesium": {...} }
      - [{ "name": "Vitamin C", ...}, {"nutrient":"Magnesium", ...}]
      - [{ "vitamin_c": {...} }, {"magnesium": {...}}]   # single-key dict items
      - { "vitamin_c_nrv": 1250, "magnesium": {...} }    # suffix noise
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            base = strip_nrv_suffixes(k)
            yield (base or k), v
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                name = item.get("name") or item.get("nutrient") or item.get("component")
                if not name:
                    # single-key dict? use that key as the name
                    if len(item) == 1:
                        k = next(iter(item))
                        base = strip_nrv_suffixes(k)
                        yield (base or k), item[k]
                        continue
                    # otherwise, try to find a key that looks like a vitamin/mineral
                    found_key = None
                    for k in item.keys():
                        if _is_vitmin_name(k):
                            found_key = k
                            break
                    if found_key:
                        base = strip_nrv_suffixes(found_key)
                        yield (base or found_key), item[found_key]
                        continue
                    # last resort
                    yield "unknown", item
                else:
                    yield str(name), item


def extract_nrv_from_nutritional_json(
    nutritional_info: Any,
) -> Tuple[bool, List[Dict[str, Any]], bool, List[str]]:
    """
    Returns:
      - has_any_nrv_percent (bool): found any numeric % that looks like NRV?
      - nrv_entries (list of {nutrient, nrv_percent})
      - has_vitamin_mineral (bool): any vitamin/mineral name detected?
      - vitmin_names (list)
    """
    data = nutritional_info
    if isinstance(nutritional_info, str):
        try:
            data = json.loads(nutritional_info)
        except Exception:
            text = nutritional_info
            raw_perc = [float(m.group(1)) for m in NRV_REGEX.finditer(text)]
            return (
                len(raw_perc) > 0,
                (
                    [{"nutrient": "unknown", "nrv_percent": max(raw_perc)}]
                    if raw_perc
                    else []
                ),
                False,
                [],
            )

    nrv_entries: List[Dict[str, Any]] = []
    vitmin_names: List[str] = []
    found_percent_any = False
    found_vitmin_any = False

    if isinstance(data, (dict, list)):
        for name, block in _iter_nutrients(data):
            is_vitmin = _is_vitmin_name(name)
            if is_vitmin and name not in vitmin_names:
                vitmin_names.append(name)
                found_vitmin_any = True

            percents: List[float] = []
            if isinstance(block, dict):
                for k, v in block.items():
                    lk = normalize_label(k)
                    if any(h in lk for h in NRV_KEY_HINTS) or "%" in lk:
                        num = _maybe_number(v)
                        if num is not None:
                            percents.append(num)
            if not percents:
                percents = _scan_for_percents(block)

            # If the "name" wasn't recognised but keys inside the block are vitamin-like,
            # treat it as a vitamin for %NRV presence purposes.
            vitaminish_inside = False
            if not is_vitmin and isinstance(block, dict):
                for k in block.keys():
                    if _is_vitmin_name(k):
                        vitaminish_inside = True
                        vn = strip_nrv_suffixes(k)
                        if vn and vn not in vitmin_names:
                            vitmin_names.append(vn)
                            found_vitmin_any = True
                is_vitmin = is_vitmin or vitaminish_inside

            if percents and is_vitmin:
                found_percent_any = True
                nrv_entries.append(
                    {
                        "nutrient": (
                            name
                            if name != "unknown"
                            else (vitmin_names[-1] if vitmin_names else "unknown")
                        ),
                        "nrv_percent": max(percents),
                    }
                )
        return found_percent_any, nrv_entries, found_vitmin_any, vitmin_names

    return False, [], False, []


# ------------------------------------------------------------------------------
# Column mapping (expects your schema, tolerates others)
# ------------------------------------------------------------------------------


def infer_semantic_fields(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols_lower = {c.lower(): c for c in df.columns}
    sku_col = cols_lower.get("sku")
    sku_name_col = (
        cols_lower.get("sku_name")
        or cols_lower.get("name")
        or cols_lower.get("product name")
    )
    nutritional_col = cols_lower.get("nutritional_info") or cols_lower.get("nutrition")

    if sku_col and sku_name_col and nutritional_col:
        return {
            "sku_col": sku_col,
            "name_col": sku_name_col,
            "nutrition_col": nutritional_col,
            "category_col": None,
            "ingredients_col": None,
            "claims_col": None,
            "is_json_nutrition": True,
        }

    def pick(candidates: List[str]) -> Optional[str]:
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in candidates):
                return c
        return None

    return {
        "sku_col": pick(["sku", "id", "code"]),
        "name_col": pick(["sku_name", "product", "name", "title"]),
        "nutrition_col": pick(
            ["nutritional_info", "nutrition", "nutrient", "composition"]
        ),
        "category_col": pick(["category", "subcat", "department", "class"]),
        "ingredients_col": pick(["ingredient"]),
        "claims_col": pick(
            ["claim", "benefit", "front of pack", "fop", "pack copy", "description"]
        ),
        "is_json_nutrition": True,
    }


# ------------------------------------------------------------------------------
# Rules engine + AI assist
# ------------------------------------------------------------------------------


def rule_requires_nrv(
    row: pd.Series,
    name_col: Optional[str],
    category_col: Optional[str],
    nutrition_col: Optional[str],
) -> Tuple[bool, str]:
    """
    Core rules — conservative and auditable:
      - Non-food-looking names => False
      - If product name implies vitamins/minerals => True
      - If nutrition JSON/text mentions vitamins/minerals => True
      - Else False (AI can upgrade to True in edge cases)
    """
    name = normalise_text(row.get(name_col, "")) if name_col else ""
    cat = normalise_text(row.get(category_col, "")) if category_col else ""
    nut_raw = row.get(nutrition_col, None) if nutrition_col else None

    name_or_cat = f"{name} {cat}".strip()
    if looks_non_food(name_or_cat):
        return False, "Non-food/cosmetic heuristic"

    if any(
        k in name
        for k in ["multivit", "vitamin", "mineral", "complex", "chelated", "zma"]
    ):
        return True, "Product name indicates vitamins/minerals"

    nut_str = stringify_maybe_json(nut_raw)
    if mentions_vitmin(nut_str):
        return True, "Nutrition mentions vitamin/mineral"

    return False, "No vitamin/mineral signals found"


def openai_client(api_key_override: Optional[str] = None) -> Optional["OpenAI"]:
    if not _openai_available:
        return None
    api_key = (
        api_key_override
        or os.getenv("OPENAI_API_KEY")
        or st.secrets.get("OPENAI_API_KEY", None)
    )
    if not api_key:
        return None
    os.environ["OPENAI_API_KEY"] = api_key
    try:
        return OpenAI()
    except Exception:
        return None


# Prompt builder reused for projection & runtime — JSON-safe
def _build_prompt(record: Dict[str, Any]) -> str:
    safe_record = make_json_safe(record)
    return f"""
You are a UK/EU food supplement labelling compliance assistant.
Decide if the product MUST display %NRV for vitamins/minerals.
Answer strictly as JSON with keys:
  - requires_nrv: boolean
  - rationale: short string (<= 240 chars)
Rules of thumb:
  • If the product is a non-food (cosmetic/device/accessory), requires_nrv = false.
  • If it's a food supplement and includes vitamins/minerals with established NRVs
    (A, D, E, K, C, B1, B2, Niacin, B6, Folate, B12, Biotin, Pantothenic acid,
     Potassium, Chloride, Calcium, Phosphorus, Magnesium, Iron, Zinc, Copper,
     Manganese, Fluoride, Selenium, Chromium, Molybdenum, Iodine), requires_nrv = true.
  • If vitamins/minerals are implied by name or claims (e.g., 'Multivitamin', 'Vitamin C 1000mg'),
    requires_nrv = true.
  • If information is insufficient, be conservative and set requires_nrv = false.

Record (JSON):
{json.dumps(safe_record, ensure_ascii=False)}
""".strip()


def llm_requires_nrv(
    client: "OpenAI", model: str, record: Dict[str, Any]
) -> Tuple[Optional[bool], str, Optional[int], Optional[int]]:
    prompt = _build_prompt(record)
    try:
        resp = client.responses.create(model=model, input=prompt, temperature=0)

        text = getattr(resp, "output_text", None)
        if not text:
            try:
                text = resp.output[0].content[0].text
            except Exception:
                text = None

        # usage may be dict or object
        in_tok = out_tok = None
        usage = getattr(resp, "usage", None)
        if usage is not None:
            if isinstance(usage, dict):
                in_tok = usage.get("input_tokens")
                out_tok = usage.get("output_tokens")
            else:
                in_tok = getattr(usage, "input_tokens", None)
                out_tok = getattr(usage, "output_tokens", None)

        if not text:
            return None, "", in_tok, out_tok

        data = json.loads(text)
        req = data.get("requires_nrv", None)
        rat = data.get("rationale", "")
        if isinstance(req, bool):
            return req, rat, in_tok, out_tok
        return None, rat, in_tok, out_tok
    except Exception:
        return None, "", None, None


# ------------------------------------------------------------------------------
# Pricing (auto) — USD per 1M tokens; converted to per-token in code
# Update values if OpenAI pricing changes.
# ------------------------------------------------------------------------------

_OPENAI_MODEL_PRICING_PER_M: Dict[str, Tuple[float, float]] = {
    # GPT-5 family
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5": (1.25, 10.00),
    # GPT-4o family
    "gpt-4o-mini": (0.60, 2.40),
    "gpt-4o": (5.00, 20.00),
    # Add more as needed
}


def get_rates_for_model(model_name: str) -> Optional[Tuple[float, float]]:
    """
    Returns (input_rate_per_token, output_rate_per_token) in USD/token.
    Matches by prefix (e.g., 'gpt-4o-2024-05-13' -> 'gpt-4o').
    """
    m = (model_name or "").lower().strip()
    for prefix, (in_per_m, out_per_m) in _OPENAI_MODEL_PRICING_PER_M.items():
        if m.startswith(prefix):
            return in_per_m / 1_000_000.0, out_per_m / 1_000_000.0
    return None


# ------------------------------------------------------------------------------
# Robust file reading (with helpful errors)
# ------------------------------------------------------------------------------


def _read_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx"):
        try:
            return pd.read_excel(uploaded_file, engine="openpyxl")
        except ImportError:
            raise RuntimeError(
                "Missing dependency 'openpyxl'. Install: pip install openpyxl"
            )
    if name.endswith(".xls"):
        try:
            return pd.read_excel(uploaded_file, engine="xlrd")
        except ImportError:
            raise RuntimeError(
                "Missing dependency 'xlrd' for .xls files. Install: pip install xlrd"
            )
    if name.endswith(".xlsb"):
        try:
            return pd.read_excel(uploaded_file, engine="pyxlsb")
        except ImportError:
            raise RuntimeError(
                "Missing dependency 'pyxlsb' for .xlsb files. Install: pip install pyxlsb"
            )
    raise RuntimeError(
        "Unsupported file type. Please upload .csv, .xlsx, .xls, or .xlsb"
    )


# ------------------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------------------


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Designed for files with columns: `sku`, `sku_name`, and `nutritional_info` (JSON)."
    )

    with st.sidebar:
        st.header("Controls")
        st.markdown("**OpenAI authentication**")
        api_key_ui = st.text_input(
            "API key (session-only)",
            type="password",
            help="Stored in session_state for this session; not persisted by the app.",
        )
        if api_key_ui:
            st.session_state["OPENAI_API_KEY_UI"] = api_key_ui

        use_ai_toggle = st.toggle(
            "Use OpenAI-assisted classification", value=AI_ENABLED_DEFAULT
        )
        model = st.text_input("OpenAI model", value=DEFAULT_MODEL)
        max_rows_ai = st.number_input(
            "AI checks — max rows",
            min_value=1,
            max_value=5000,
            value=AI_MAX_ROWS_DEFAULT,
            step=10,
        )

        st.divider()
        st.markdown("**Projected cost uses built-in rates for the selected model.**")

        st.divider()
        st.markdown(
            "**Expected schema**: `sku`, `sku_name`, `nutritional_info` (JSON). "
            "App also handles broader schemas on a best-effort basis."
        )

    uploaded = st.file_uploader(
        "Upload CSV or Excel", type=["csv", "xlsx", "xls", "xlsb"]
    )
    if not uploaded:
        st.info("Upload a product file to preview and enable the Run button.")
        st.stop()

    # Read file (safe; no processing starts yet)
    try:
        df = _read_table(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    if df.empty:
        st.warning("The file is empty.")
        st.stop()

    st.subheader("Source Preview")
    st.dataframe(df.head(25), use_container_width=True)

    # Map columns
    fields = infer_semantic_fields(df)
    sku_col = fields["sku_col"]
    name_col = fields["name_col"]
    nutrition_col = fields["nutrition_col"]
    category_col = fields["category_col"]
    is_json_nutrition = fields["is_json_nutrition"]

    st.markdown("#### Column Mapping")
    st.write(
        {
            "SKU": sku_col,
            "Product Name": name_col,
            "Nutrition (JSON)": nutrition_col,
            "Category": category_col,
            "JSON expected": is_json_nutrition,
        }
    )

    total_rows = len(df)

    # === PROJECTION (toggle only; independent of API key availability) ===
    planned_ai_calls = min(total_rows, int(max_rows_ai)) if use_ai_toggle else 0

    # Estimate tokens per call from real prompt+row sample (first N rows)
    est_output_tokens_per_call = 60
    est_input_tokens_total = 0
    sample_rows = min(planned_ai_calls, total_rows)
    for i in range(sample_rows):
        r = df.iloc[i]
        record = {
            "sku": r.get(sku_col) if sku_col else None,
            "product_name": r.get(name_col) if name_col else None,
            "nutrition_excerpt": (
                str(r.get(nutrition_col))[:800] if nutrition_col else None
            ),
        }
        prompt = _build_prompt(record)
        est_input_tokens_total += approx_tokens_from_text(prompt)

    if sample_rows and planned_ai_calls > sample_rows:
        scale = planned_ai_calls / sample_rows
        est_input_tokens_total = int(est_input_tokens_total * scale)

    est_output_tokens_total = planned_ai_calls * est_output_tokens_per_call

    rates = get_rates_for_model(model)
    if rates:
        in_rate_per_tok, out_rate_per_tok = rates
        projected_cost = (
            est_input_tokens_total * in_rate_per_tok
            + est_output_tokens_total * out_rate_per_tok
        )
        pricing_note = f"Auto-priced using '{model}' rates."
    else:
        projected_cost = 0.0
        pricing_note = (
            "Unknown pricing for this model. Update the pricing map in code or choose "
            "gpt-5 / gpt-5-mini / gpt-5-nano / gpt-4o / gpt-4o-mini."
        )

    st.markdown("#### Cost Projection (pre-run)")
    st.write(
        {
            "Model": model,
            "Planned AI calls": planned_ai_calls,
            "Estimated input tokens (total)": int(est_input_tokens_total),
            "Estimated output tokens (total)": int(est_output_tokens_total),
            "Projected cost (USD)": round(projected_cost, 6),
            "Pricing note": pricing_note,
        }
    )

    # --- Manual trigger to start the audit ---
    run_clicked = st.button("▶️ Run NRV Audit", type="primary")
    if not run_clicked:
        st.stop()

    # ---------------------- RUN AUDIT (on click) ----------------------

    # Runtime AI enablement depends on key availability
    use_ai_runtime = use_ai_toggle
    client = None
    if use_ai_runtime:
        api_key_override = st.session_state.get("OPENAI_API_KEY_UI")
        client = openai_client(api_key_override)
        if client is None:
            st.warning(
                "OpenAI key not configured or SDK unavailable. The audit will run rule-only."
            )
            use_ai_runtime = False

    # Progress bar
    progress = st.progress(0)
    progress_text = st.empty()

    # KPI counters (internal only)
    kpi_must = 0
    kpi_present = 0
    kpi_fail = 0

    results: List[Dict[str, Any]] = []
    ai_calls = 0
    actual_in_tokens = 0
    actual_out_tokens = 0

    for idx, row in df.iterrows():
        progress.progress(int(((idx + 1) / total_rows) * 100))
        progress_text.text(f"Processing row {idx + 1} of {total_rows} ...")

        # Rule decision
        rule_req, rule_reason = rule_requires_nrv(
            row,
            name_col=name_col,
            category_col=category_col,
            nutrition_col=nutrition_col,
        )

        # JSON extraction
        json_has_nrv, json_nrv_entries, json_has_vitmin, json_vitmins = (
            extract_nrv_from_nutritional_json(
                row.get(nutrition_col) if nutrition_col else None
            )
        )

        requires_by_json = json_has_vitmin
        requires = bool(requires_by_json or rule_req)

        # AI assist (optional)
        ai_req, ai_rat = (None, "")
        ai_reason = None
        if use_ai_runtime and ai_calls < int(max_rows_ai) and client is not None:
            record = {
                "sku": row.get(sku_col) if sku_col else None,
                "product_name": row.get(name_col) if name_col else None,
                "nutrition_top": json_nrv_entries[:10],
                "vitmins_detected": json_vitmins[:10],
            }
            req, rat, in_tok, out_tok = llm_requires_nrv(client, model, record)
            ai_req, ai_rat = req, rat
            ai_calls += 1
            if isinstance(in_tok, int):
                actual_in_tokens += in_tok
            if isinstance(out_tok, int):
                actual_out_tokens += out_tok
            if ai_req is True:
                requires = True
            if ai_req is None:
                ai_reason = "no_response_or_parse_error"
        else:
            if not use_ai_runtime:
                ai_reason = "disabled"
            elif client is None:
                ai_reason = "no_api_key_or_sdk"
            elif ai_calls >= int(max_rows_ai):
                ai_reason = "capped_by_max_rows"

        # Update KPIs (internal)
        if requires:
            kpi_must += 1
            if not json_has_nrv:
                kpi_fail += 1
        if json_has_nrv:
            kpi_present += 1

        # Build debug summary (single column)
        debug_payload: Dict[str, Any] = {
            "rule": {"requires": rule_req, "reason": rule_reason},
            "json": {
                "vitamins_detected": json_vitmins,
                "has_nrv_percent": json_has_nrv,
            },
        }
        if use_ai_toggle:  # show AI section whenever the toggle is on
            debug_payload["ai"] = {
                "requires": ai_req,
                "rationale": ai_rat,
                "reason": ai_reason,
            }

        debug_checks = json.dumps(make_json_safe(debug_payload), ensure_ascii=False)

        # Store ONLY the requested columns for the table
        results.append(
            {
                "_row": idx,
                "sku": row.get(sku_col) if sku_col else None,
                "sku_name": row.get(name_col) if name_col else None,
                "needs_nrv": requires,
                "debug_checks": debug_checks,
            }
        )

    progress.progress(100)
    progress_text.text("Processing complete.")

    # Build audited table: original df + new columns (no duplicated extras)
    out = pd.DataFrame(results)
    out_for_join = out.drop(columns=["_row", "sku", "sku_name"], errors="ignore")
    audited = pd.concat([df, out_for_join], axis=1)
    if audited.columns.duplicated().any():
        audited = audited.loc[:, ~audited.columns.duplicated()]

    st.subheader("Audit Output")
    st.dataframe(audited.head(50), use_container_width=True)

    # Actual cost using returned usage + auto rates
    actual_cost = None
    if actual_in_tokens or actual_out_tokens:
        rates2 = get_rates_for_model(model)
        if rates2:
            in_rate_per_tok, out_rate_per_tok = rates2
            actual_cost = (
                actual_in_tokens * in_rate_per_tok
                + actual_out_tokens * out_rate_per_tok
            )

    st.markdown("#### Summary KPIs")
    kpis = {
        "Rows audited": len(df),
        "Require %NRV (needs_nrv=True)": int(kpi_must),
        "NRV present (JSON detected any %)": int(kpi_present),
        "Open findings (needs_nrv=True but no % found)": int(kpi_fail),
        "AI calls used": ai_calls,
        "Estimated input tokens (pre-run)": int(est_input_tokens_total),
        "Estimated output tokens (pre-run)": int(est_output_tokens_total),
        "Projected cost (USD)": round(projected_cost, 6),
        "Actual input tokens": actual_in_tokens,
        "Actual output tokens": actual_out_tokens,
        "Actual cost (USD)": (
            round(actual_cost, 6) if actual_cost is not None else "n/a"
        ),
    }
    st.write(kpis)

    # Export
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        audited.to_excel(writer, index=False, sheet_name="NRV_Audit")
    st.download_button(
        "⬇️ Download audited Excel",
        data=buf.getvalue(),
        file_name="nrv_audit_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.info(
        "The table adds only 'needs_nrv' and 'debug_checks'. "
        "Auto-pricing uses built-in rates; projection estimates input tokens (~4 chars/token) "
        "and assumes ~60 output tokens/call. Actual cost uses API usage if available. "
        "For .xlsx uploads, ensure 'openpyxl' is installed."
    )


if __name__ == "__main__":
    main()
