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
DEFAULT_MODEL = "gpt-5-mini"  # set any Responses-capable text model
AI_ENABLED_DEFAULT = True
AI_MAX_ROWS_DEFAULT = 500

# --- Canonical EU/UK NRV nutrients for vitamins & minerals --------------------
# (Presence of these implies product should carry %NRV)
VITAMIN_MINERAL_NRV_TERMS = {
    # Vitamins
    "vitamin a",
    "retinol",
    "beta-carotene",
    "vitamin d",
    "cholecalciferol",
    "ergocalciferol",
    "vitamin e",
    "tocopherol",
    "alpha-tocopherol",
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


def looks_non_food(name_or_category: str) -> bool:
    blob = (name_or_category or "").lower()
    return any(tok in blob for tok in NON_FOOD_TOKENS)


def mentions_vitmin(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(term in t for term in VITAMIN_MINERAL_NRV_TERMS)


def stringify_maybe_json(val: Any) -> str:
    if isinstance(val, (dict, list)):
        try:
            return json.dumps(val, ensure_ascii=False).lower()
        except Exception:
            return str(val).lower()
    return normalise_text(val)


# ------------------------------------------------------------------------------
# JSON parsing for nutritional_info
# ------------------------------------------------------------------------------


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


NRV_KEY_HINTS = {
    "nrv",
    "nrv%",
    "%nrv",
    "nrv_percent",
    "percent_nrv",
    "daily",
    "ri",
    "reference intake",
    "rda",
}


def _is_vitmin_name(n: str) -> bool:
    n = n.lower().strip().replace("-", " ")
    return mentions_vitmin(n)


def _scan_for_percents(obj: Any) -> List[float]:
    """Extract plausible % values from nested objects (dict/list/str/num)."""
    found: List[float] = []

    def add_if_number(v: Any):
        num = _maybe_number(v)
        if num is not None:
            found.append(num)

    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
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
    Yield (nutrient_name, value_block) pairs from common shapes:
      - { "Vitamin C": {"amount": 1000, "nrv_percent": 1250}, ... }
      - [{ "name": "Vitamin C", "nrv_percent": 1250, ...}, ...]
      - { "vitamin_c_nrv": 1250, "magnesium": { ... } }
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k), v
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                name = item.get("name") or item.get("nutrient") or item.get("component")
                if name:
                    yield str(name), item
                else:
                    yield "unknown", item


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
            # Not JSON; fall back to regex detection
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
                    lk = str(k).lower()
                    if any(h in lk for h in NRV_KEY_HINTS) or "%" in lk:
                        num = _maybe_number(v)
                        if num is not None:
                            percents.append(num)
            if not percents:
                percents = _scan_for_percents(block)

            if percents and is_vitmin:
                found_percent_any = True
                nrv_entries.append({"nutrient": name, "nrv_percent": max(percents)})

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

    # Fallback for broader schemas
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


def llm_requires_nrv(
    client: "OpenAI", model: str, record: Dict[str, Any]
) -> Tuple[Optional[bool], str, Optional[int], Optional[int]]:
    prompt = f"""
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
{json.dumps(record, ensure_ascii=False)}
"""
    try:
        resp = client.responses.create(model=model, input=prompt, temperature=0)

        # Extract text robustly
        text = getattr(resp, "output_text", None)
        if not text:
            try:
                text = resp.output[0].content[0].text
            except Exception:
                text = None

        # Extract usage robustly (SDK may return attrs or dict)
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
                "Missing dependency 'openpyxl'. Install it with: pip install openpyxl"
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
        # Session-only API key
        st.markdown("**OpenAI authentication**")
        api_key_ui = st.text_input(
            "API key (session-only)",
            type="password",
            help="Stored in session_state for this session; not persisted by the app.",
        )
        if api_key_ui:
            st.session_state["OPENAI_API_KEY_UI"] = api_key_ui

        use_ai = st.toggle(
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
        st.markdown("**Pricing (USD per 1K tokens)**")
        price_in = st.number_input(
            "Input rate ($/1K tokens)",
            min_value=0.0,
            value=0.0,
            step=0.001,
            format="%.4f",
        )
        price_out = st.number_input(
            "Output rate ($/1K tokens)",
            min_value=0.0,
            value=0.0,
            step=0.001,
            format="%.4f",
        )
        st.caption(
            "Enter your contract rates here to enable cost projection & actuals."
        )

        st.markdown("**Projection assumptions (tokens per AI call)**")
        assumed_in_toks = st.number_input(
            "Assumed input tokens per call", min_value=0, value=600, step=50
        )
        assumed_out_toks = st.number_input(
            "Assumed output tokens per call", min_value=0, value=120, step=10
        )

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

    # Client preload (no calls yet)
    client = None
    if use_ai:
        api_key_override = st.session_state.get("OPENAI_API_KEY_UI")
        client = openai_client(api_key_override)
        if client is None:
            st.warning(
                "OpenAI key not configured or SDK unavailable. The audit will run rule-only."
            )
            use_ai = False

    total_rows = len(df)
    planned_ai_calls = min(total_rows, int(max_rows_ai)) if use_ai else 0

    # Projected cost for this run (pre-run)
    projected_input_tokens = planned_ai_calls * assumed_in_toks
    projected_output_tokens = planned_ai_calls * assumed_out_toks
    projected_cost = (projected_input_tokens / 1000.0) * price_in + (
        projected_output_tokens / 1000.0
    ) * price_out

    st.markdown("#### Cost Projection (pre-run)")
    st.write(
        {
            "Planned AI calls": planned_ai_calls,
            "Projected input tokens": projected_input_tokens,
            "Projected output tokens": projected_output_tokens,
            "Projected cost (USD)": round(projected_cost, 6),
        }
    )

    # --- Manual trigger to start the audit ---
    run_clicked = st.button("▶️ Run NRV Audit", type="primary")

    if not run_clicked:
        st.stop()

    # ---------------------- RUN AUDIT (on click) ----------------------

    # Progress bar
    progress = st.progress(0)
    progress_text = st.empty()

    results = []
    ai_calls = 0

    # Actual usage trackers (if API returns usage)
    actual_in_tokens = 0
    actual_out_tokens = 0

    for idx, row in df.iterrows():
        # Update progress UI (0..100)
        progress.progress(int(((idx + 1) / total_rows) * 100))
        progress_text.text(f"Processing row {idx + 1} of {total_rows} ...")

        # Core rule decision (name/category/nutrition text)
        rule_req, rule_reason = rule_requires_nrv(
            row,
            name_col=name_col,
            category_col=category_col,
            nutrition_col=nutrition_col,
        )

        # JSON-based extraction (primary for your schema)
        json_has_nrv, json_nrv_entries, json_has_vitmin, json_vitmins = (
            extract_nrv_from_nutritional_json(
                row.get(nutrition_col) if nutrition_col else None
            )
        )

        requires_by_json = json_has_vitmin
        requires = bool(requires_by_json or rule_req)

        # AI assist (optional, budget-capped)
        ai_req, ai_rat = (None, "")
        if use_ai and ai_calls < int(max_rows_ai):
            record = {
                "sku": row.get(sku_col) if sku_col else None,
                "product_name": row.get(name_col) if name_col else None,
                "nutrition_top": json_nrv_entries[:10],
                "vitmins_detected": json_vitmins[:10],
            }
            req, rat, in_tok, out_tok = llm_requires_nrv(client, model, record)
            ai_req, ai_rat = req, rat
            ai_calls += 1

            # Accumulate actual usage if provided by API
            if isinstance(in_tok, int):
                actual_in_tokens += in_tok
            if isinstance(out_tok, int):
                actual_out_tokens += out_tok

            requires = bool(requires or (ai_req is True))

        # Presence check: True if JSON had at least one vitamin/mineral with a numeric %.
        has_nrv_value = bool(json_has_nrv)

        status = "PASS" if (not requires or (requires and has_nrv_value)) else "FAIL"

        results.append(
            {
                "_row": idx,
                "sku": row.get(sku_col) if sku_col else None,
                "sku_name": row.get(name_col) if name_col else None,
                "requires_nrv_rule": rule_req,
                "rule_rationale": rule_reason,
                "requires_nrv_json": requires_by_json,
                "vitmins_detected_json": (
                    ", ".join(json_vitmins) if json_vitmins else ""
                ),
                "nrv_entries_json": (
                    "; ".join(
                        f"{e['nutrient']}: {e['nrv_percent']}%"
                        for e in json_nrv_entries
                    )
                    if json_nrv_entries
                    else ""
                ),
                "requires_nrv_ai": ai_req,
                "ai_rationale": ai_rat,
                "requires_nrv_final": requires,
                "has_nrv_value": has_nrv_value,
                "status": status,
            }
        )

    # Finalize progress
    progress.progress(100)
    progress_text.text("Processing complete.")

    out = pd.DataFrame(results)

    # ---- Avoid duplicate column names when joining with original df ----
    out_for_join = out.drop(columns=["_row", "sku", "sku_name"], errors="ignore")
    audited = pd.concat([df, out_for_join], axis=1)

    # Belt-and-braces: drop any remaining duplicate columns (keep first)
    if audited.columns.duplicated().any():
        audited = audited.loc[:, ~audited.columns.duplicated()]

    st.subheader("Audit Output")
    st.dataframe(audited.head(50), use_container_width=True)

    # Actual cost computation (only if usage available & pricing entered)
    actual_cost = None
    if (actual_in_tokens or actual_out_tokens) and (price_in > 0 or price_out > 0):
        actual_cost = (actual_in_tokens / 1000.0) * price_in + (
            actual_out_tokens / 1000.0
        ) * price_out

    st.markdown("#### Summary KPIs")
    total = len(audited)
    must = int(audited["requires_nrv_final"].sum())
    present = int(audited["has_nrv_value"].sum())
    fail = int((audited["requires_nrv_final"] & ~audited["has_nrv_value"]).sum())

    kpis = {
        "Rows audited": total,
        "Require %NRV": must,
        "NRV present (any vitamin/mineral)": present,
        "Open findings (fail)": fail,
        "AI calls used": ai_calls,
        "Projected cost (USD)": round(projected_cost, 6),
        "Actual input tokens": actual_in_tokens,
        "Actual output tokens": actual_out_tokens,
    }
    if actual_cost is not None:
        kpis["Actual cost (USD)"] = round(actual_cost, 6)
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
        "Security: The API key you paste is kept only in Streamlit session_state for this session and is not persisted by the app. "
        "Set token pricing in the sidebar to enable cost projection & actuals. "
        "For .xlsx uploads, ensure 'openpyxl' is installed."
    )


if __name__ == "__main__":
    main()
