import os
import re
import io
import json
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import streamlit as st

# OpenAI SDK (Responses API)
try:
    from openai import OpenAI

    _openai_available = True
except Exception:
    _openai_available = False

APP_TITLE = "NRV Audit — Product Label Compliance (EU/UK)"
DEFAULT_MODEL = "gpt-5-mini"  # editable in the sidebar

# --- Canonical EU/UK NRV nutrients for vitamins & minerals ---------------
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

NON_FOOD_CATEGORIES = {
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

NRV_REGEX = re.compile(r"(\d+(\.\d+)?)\s*%?\s*NRV", flags=re.IGNORECASE)


def normalise_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def looks_non_food(category_text: str, product_name: str) -> bool:
    blob = f"{category_text} {product_name}".lower()
    return any(tok in blob for tok in NON_FOOD_CATEGORIES)


def mentions_vitmin(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    return any(term in t for term in VITAMIN_MINERAL_NRV_TERMS)


def infer_semantic_fields(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Heuristically infer columns: product name, category, ingredients, nutrition, claims."""

    def pick(candidates: List[str]) -> Optional[str]:
        for c in df.columns:
            lc = c.lower()
            if any(k in lc for k in candidates):
                return c
        return None

    return {
        "name_col": pick(["product", "name", "title"]),
        "category_col": pick(["category", "subcat", "department", "class"]),
        "ingredients_col": pick(["ingredient"]),
        "nutrition_col": pick(["nutrition", "nutrient", "composition"]),
        "claims_col": pick(
            ["claim", "benefit", "front of pack", "fop", "pack copy", "description"]
        ),
    }


def row_has_nrv_value(
    row: pd.Series, nrv_columns: List[str], free_text_fields: List[str]
) -> Tuple[bool, List[str], List[str]]:
    cols_hit, text_hits = [], []
    # Check explicit NRV columns
    for c in nrv_columns:
        val = row.get(c, None)
        if pd.notna(val) and str(val).strip() not in ("", "0", "0.0"):
            cols_hit.append(c)
    # Check free text for "xx% NRV"
    for f in free_text_fields:
        txt = normalise_text(row.get(f, ""))
        if txt and NRV_REGEX.search(txt):
            text_hits.append(f)
    return bool(cols_hit or text_hits), cols_hit, text_hits


def detect_nrv_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if "nrv" in c.lower()]


def rule_requires_nrv(
    row: pd.Series,
    name_col: Optional[str],
    category_col: Optional[str],
    ingredients_col: Optional[str],
    nutrition_col: Optional[str],
    claims_col: Optional[str],
) -> Tuple[bool, str]:
    """Conservative rule set:
    - If clearly non-food (cosmetic/device/etc.): False
    - Else if ingredients/nutrition/claims mention vitamins/minerals from canonical list: True
    - Else if name suggests multivitamin/mineral/complex: True
    - Else: False (LLM can upgrade to True if necessary)
    """
    name = normalise_text(row.get(name_col, "")) if name_col else ""
    cat = normalise_text(row.get(category_col, "")) if category_col else ""
    ing = normalise_text(row.get(ingredients_col, "")) if ingredients_col else ""
    nut = normalise_text(row.get(nutrition_col, "")) if nutrition_col else ""
    cla = normalise_text(row.get(claims_col, "")) if claims_col else ""

    if looks_non_food(cat, name):
        return False, "Non-food/cosmetic category heuristic"

    # Strong name cues
    if any(
        k in name
        for k in ["multivit", "vitamin", "mineral", "complex", "chelated", "zma"]
    ):
        return True, "Product name indicates vitamins/minerals"

    # Signals in ingredients/nutrition/claims
    if mentions_vitmin(" ".join([ing, nut, cla])):
        return True, "Ingredients/Nutrition/Claims mention vitamin/mineral"

    return False, "No vitamin/mineral signals found"


# === OpenAI client loader with UI-provided API key =======================
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
    os.environ["OPENAI_API_KEY"] = api_key  # ensures SDK can pick it up
    try:
        return OpenAI()
    except Exception:
        return None


def llm_requires_nrv(
    client: "OpenAI", model: str, record: Dict[str, Any]
) -> Tuple[Optional[bool], str]:
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
        resp = client.responses.create(
            model=model,
            input=prompt,
            temperature=0,
        )
        text = getattr(resp, "output_text", None)
        if not text:
            try:
                text = resp.output[0].content[0].text
            except Exception:
                text = None
        if not text:
            return None, ""
        data = json.loads(text)
        req = data.get("requires_nrv", None)
        rat = data.get("rationale", "")
        if isinstance(req, bool):
            return req, rat
        return None, rat
    except Exception:
        return None, ""


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Upload → classify NRV obligation → verify NRV presence → export.")

    with st.sidebar:
        st.header("Controls")
        # API key input (session-only)
        st.markdown("**OpenAI authentication**")
        api_key_ui = st.text_input(
            "API key (session-only)",
            type="password",
            help="Stored in session_state only for this browser session; not persisted to disk.",
        )
        if api_key_ui:
            st.session_state["OPENAI_API_KEY_UI"] = api_key_ui

        use_ai = st.toggle("Use OpenAI-assisted classification", value=True)
        model = st.text_input("OpenAI model", value=DEFAULT_MODEL)
        max_rows_ai = st.number_input(
            "AI checks — max rows", min_value=1, max_value=5000, value=500, step=10
        )
        st.caption("Tip: Leave API key blank to use environment or Streamlit secrets.")
        st.divider()
        st.markdown(
            "**Expected columns (flexible):** Product Name, Category, Ingredients, Nutrition, Claims, and any *NRV* columns."
        )

    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if not uploaded:
        st.info("Upload a product file to get started.")
        st.stop()

    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    if df.empty:
        st.warning("The file is empty.")
        st.stop()

    st.subheader("Source Preview")
    st.dataframe(df.head(25), use_container_width=True)

    fields = infer_semantic_fields(df)
    name_col = fields["name_col"]
    category_col = fields["category_col"]
    ingredients_col = fields["ingredients_col"]
    nutrition_col = fields["nutrition_col"]
    claims_col = fields["claims_col"]

    nrv_cols = detect_nrv_columns(df)
    free_text_fields = [c for c in [ingredients_col, nutrition_col, claims_col] if c]

    st.markdown("#### Column Mapping")
    st.write(
        {
            "Product Name": name_col,
            "Category": category_col,
            "Ingredients": ingredients_col,
            "Nutrition": nutrition_col,
            "Claims": claims_col,
            "Detected NRV columns": nrv_cols,
        }
    )

    # Build OpenAI client using UI key (if provided) else env/secrets
    client = None
    if use_ai:
        api_key_override = st.session_state.get("OPENAI_API_KEY_UI")
        client = openai_client(api_key_override)
        if client is None:
            st.warning(
                "OpenAI key not configured or SDK not available. Falling back to rule-only checks."
            )
            use_ai = False

    results = []
    ai_calls = 0
    for idx, row in df.iterrows():
        rule_req, rule_reason = rule_requires_nrv(
            row, name_col, category_col, ingredients_col, nutrition_col, claims_col
        )

        record = {
            "product_name": row.get(name_col) if name_col else None,
            "category": row.get(category_col) if category_col else None,
            "ingredients": row.get(ingredients_col) if ingredients_col else None,
            "nutrition": row.get(nutrition_col) if nutrition_col else None,
            "claims": row.get(claims_col) if claims_col else None,
        }

        ai_req, ai_rat = (None, "")
        if use_ai and ai_calls < max_rows_ai:
            ai_req, ai_rat = llm_requires_nrv(client, model, record)
            ai_calls += 1

        requires = bool(rule_req or (ai_req is True))
        has_nrv, cols_hit, text_hits = row_has_nrv_value(
            row, nrv_cols, free_text_fields
        )
        status = "PASS" if (not requires or (requires and has_nrv)) else "FAIL"

        results.append(
            {
                "_row": idx,
                "requires_nrv_rule": rule_req,
                "rule_rationale": rule_reason,
                "requires_nrv_ai": ai_req,
                "ai_rationale": ai_rat,
                "requires_nrv_final": requires,
                "has_nrv_value": has_nrv,
                "nrv_columns_hit": ", ".join(cols_hit),
                "nrv_text_hits_in_fields": ", ".join(text_hits),
                "status": status,
            }
        )

    out = pd.DataFrame(results)
    audited = pd.concat([df, out.drop(columns=["_row"])], axis=1)

    st.subheader("Audit Output")
    st.dataframe(audited.head(50), use_container_width=True)

    st.markdown("#### Summary KPIs")
    total = len(audited)
    must = int(audited["requires_nrv_final"].sum())
    present = int(audited["has_nrv_value"].sum())
    fail = int((audited["requires_nrv_final"] & ~audited["has_nrv_value"]).sum())
    st.write(
        {
            "Rows audited": total,
            "Require %NRV": must,
            "NRV present (any)": present,
            "Open findings (fail)": fail,
            "AI calls used": ai_calls,
        }
    )

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
        "Security: The key you paste is kept only in Streamlit session_state for this browser session and is not persisted to disk by the app."
    )


if __name__ == "__main__":
    main()
