import json
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Optional RDKit support (app still runs without it)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import Draw
    from rdkit import DataStructs

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


IFRA_SMILES_PATH = Path("ifra_category4_smiles.csv")
IFRA_FEATURES_PATH = Path("ifra_category4_features.csv")
WATCHLIST_PATH = Path("AI_Predictive_Watchlist.csv")
SAMPLE_FORMULA_PATH = Path("sample_formula.csv")


st.set_page_config(
    page_title="Nose What's Legal",
    page_icon="🧪",
    layout="wide",
)

st.title("Nose What's Legal")
st.caption("Regulatory analytics, structure-based screening, and formula auditing for IFRA Category 4 workflows.")


def _safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


@st.cache_data(show_spinner=False)
def load_ifra_raw() -> pd.DataFrame:
    if not IFRA_SMILES_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(IFRA_SMILES_PATH)


@st.cache_data(show_spinner=False)
def load_feature_map() -> Dict[str, float]:
    if not IFRA_FEATURES_PATH.exists():
        return {}
    feat_df = pd.read_csv(IFRA_FEATURES_PATH)
    if "ingredient_name" not in feat_df.columns or "LogP" not in feat_df.columns:
        return {}

    mapping: Dict[str, float] = {}
    for _, row in feat_df.iterrows():
        name = str(row.get("ingredient_name", "")).strip().lower()
        if not name:
            continue
        mapping[name] = _safe_float(row.get("LogP"), default=0.0)
    return mapping


@st.cache_data(show_spinner=False)
def load_directory_data() -> pd.DataFrame:
    records: List[dict] = []

    ifra_df = load_ifra_raw()
    logp_map = load_feature_map()

    if not ifra_df.empty:
        for _, row in ifra_df.iterrows():
            name = str(row.get("ingredient_name", "")).strip()
            if not name or name.lower() == "nan":
                continue

            limit_val = _safe_float(row.get("category_4_limit_percent"), default=0.0)
            status = "Banned" if limit_val == 0.0 else "Restricted"
            reason_val = row.get("reason", "Regulatory Risk")
            if pd.isna(reason_val) or str(reason_val).lower() == "nan":
                safety_risk = "Safety Risk (Unspecified)"
            else:
                safety_risk = str(reason_val)

            rule_year = row.get("rule_year")
            year_str = "Unknown" if pd.isna(rule_year) else str(int(rule_year))
            smiles_val = str(row.get("smiles", "")).strip()

            records.append(
                {
                    "Name": name,
                    "SMILES": smiles_val,
                    "Status": status,
                    "Safety_Risk": safety_risk,
                    "Limit_Cat4": limit_val,
                    "Year": year_str,
                    "Category": status,
                    "LogP": logp_map.get(name.lower(), 0.0),
                }
            )

    existing_names = {item["Name"].lower() for item in records}
    if WATCHLIST_PATH.exists():
        watch_df = pd.read_csv(WATCHLIST_PATH)
        for _, row in watch_df.iterrows():
            name = str(row.get("Candidate_Name", "")).strip()
            if not name or name.lower() == "nan" or name.lower() in existing_names:
                continue

            records.append(
                {
                    "Name": name,
                    "SMILES": str(row.get("Candidate_SMILES", "")).strip(),
                    "Status": "Safe / Unregulated (High Risk)",
                    "Safety_Risk": str(row.get("AI_Predicted_Risk", "Potential Risk")),
                    "Limit_Cat4": 100.0,
                    "Year": "N/A",
                    "Category": "Safe",
                    "LogP": 0.0,
                }
            )

    if not records:
        return pd.DataFrame(
            {
                "Name": ["Eugenol", "Lilial", "Limonene"],
                "SMILES": [
                    "COC1=C(O)C=CC(CC=C)=C1",
                    "CC(C)(C)c1ccc(CC(C)C=O)cc1",
                    "CC1=CCC(CC1)C(=C)C",
                ],
                "Status": ["Restricted", "Banned", "Unregulated"],
                "Safety_Risk": ["Skin Sensitization", "Reproductive Toxicity", "Low"],
                "Limit_Cat4": [2.5, 0.0, None],
                "Year": ["2020", "2019", "Unknown"],
                "Category": ["Restricted", "Banned", "Safe"],
                "LogP": [0.0, 0.0, 0.0],
            }
        )

    return pd.DataFrame(records)


@st.cache_data(show_spinner=False)
def build_limit_lookup() -> Dict[str, dict]:
    ifra_df = load_ifra_raw()
    lookup: Dict[str, dict] = {}

    if ifra_df.empty:
        return lookup

    for _, row in ifra_df.iterrows():
        name = str(row.get("ingredient_name", "")).strip()
        if not name or name.lower() == "nan":
            continue

        limit = _safe_float(row.get("category_4_limit_percent"), default=0.0)
        reason = str(row.get("reason", "Regulatory Risk"))

        lookup[name.lower()] = {"canonical": name, "limit": limit, "reason": reason}

        synonyms = row.get("synonyms")
        if pd.notna(synonyms):
            parts = [
                s.strip()
                for s in str(synonyms).replace(";", "|").replace("\n", "|").split("|")
                if s.strip()
            ]
            for syn in parts:
                lookup[syn.lower()] = {"canonical": name, "limit": limit, "reason": reason}

    return lookup


def parse_formula_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        formula_df = pd.read_csv(uploaded_file)
    elif SAMPLE_FORMULA_PATH.exists():
        formula_df = pd.read_csv(SAMPLE_FORMULA_PATH)
    else:
        return pd.DataFrame()

    cols = {c.lower(): c for c in formula_df.columns}
    if "ingredient" not in cols or "percentage" not in cols:
        return pd.DataFrame()

    clean = formula_df[[cols["ingredient"], cols["percentage"]]].copy()
    clean.columns = ["Ingredient", "Percentage"]
    clean["Ingredient"] = clean["Ingredient"].astype(str).str.strip()
    clean["Percentage"] = pd.to_numeric(clean["Percentage"], errors="coerce")
    clean = clean.dropna(subset=["Ingredient", "Percentage"])
    return clean


def audit_formula(formula_df: pd.DataFrame, lookup: Dict[str, dict]) -> pd.DataFrame:
    results = []
    for _, row in formula_df.iterrows():
        ing = str(row["Ingredient"]).strip()
        pct = float(row["Percentage"])
        key = ing.lower()
        item = lookup.get(key)

        if item is None:
            results.append(
                {
                    "Ingredient": ing,
                    "In_Formula_%": pct,
                    "IFRA_Limit_%": "Unregulated",
                    "Status": "PASS",
                    "Regulatory_Notes": "No IFRA Category 4 restriction found.",
                }
            )
            continue

        limit = item["limit"]
        reason = item["reason"]
        canonical = item["canonical"]

        if limit == 0.0:
            status = "FAIL"
        elif pct > limit:
            status = "FAIL"
        else:
            status = "PASS"

        results.append(
            {
                "Ingredient": ing,
                "In_Formula_%": pct,
                "IFRA_Limit_%": limit,
                "Status": status,
                "Regulatory_Notes": f"Matched as: {canonical}. Reason: {reason}",
            }
        )

    return pd.DataFrame(results)


def compute_replacements(target_smiles: str, candidate_df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    if not RDKIT_AVAILABLE or not target_smiles:
        return pd.DataFrame()

    target_mol = Chem.MolFromSmiles(target_smiles)
    if target_mol is None:
        return pd.DataFrame()

    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, nBits=1024)

    rows = []
    for _, row in candidate_df.iterrows():
        name = str(row.get("Name", "")).strip()
        smiles = str(row.get("SMILES", "")).strip()
        if not name or not smiles:
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        score = DataStructs.TanimotoSimilarity(target_fp, fp)
        rows.append(
            {
                "Candidate": name,
                "Similarity_%": round(score * 100, 1),
                "SMILES": smiles,
                "Status": row.get("Status", "Safe / Unregulated"),
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("Similarity_%", ascending=False)
    return out.head(top_k)


ODOR_FAMILY_KEYWORDS: Dict[str, List[str]] = {
    "musk":      ["musk", "ambrette", "galaxolide", "habanolide", "ambrettolide", "exaltolide", "ambroxan", "cetalox", "romandolide"],
    "vanilla":   ["vanillin", "vanilla", "ethyl vanillin", "coumarin", "heliotropin", "piperonal", "isoeugenol", "iso eugenol", "benzyl benzoate"],
    "citrus":    ["limonene", "lemon", "orange", "bergamot", "citral", "citronellol", "geranial", "neral", "grapefruit", "citronellal", "linalyl"],
    "floral":    ["rose", "jasmine", "lily", "geraniol", "linalool", "violet", "peony", "lilial", "lyral", "hedione", "phenyl ethyl", "floralozone", "benzyl alcohol"],
    "woody":     ["cedar", "sandalwood", "vetiver", "patchouli", "guaiac", "oakmoss", "timbersilk", "cashmeran"],
    "spicy":     ["eugenol", "cinnamic", "cinnamaldehyde", "clove", "cardamom", "methyl eugenol"],
    "fresh":     ["menthol", "eucalyptol", "camphor", "cineole", "mint"],
    "green":     ["violet leaf", "galbanum", "hyacinth", "hexenol"],
    "powdery":   ["orris", "iris", "ionone"],
    "fruity":    ["peach", "apricot", "raspberry", "strawberry", "apple", "cherry", "pineapple", "lychee", "mandarin"],
    "aldehydic": ["nonanal", "decanal", "undecanal", "dodecanal", "aldehyde c-"],
}

FAMILY_COLORS: Dict[str, Dict[str, str]] = {
    "musk":      {"bg": "#C8A882", "border": "#7B5B3A"},
    "vanilla":   {"bg": "#F5E6C8", "border": "#C8A000"},
    "citrus":    {"bg": "#FFE066", "border": "#CCA800"},
    "floral":    {"bg": "#FFB3D1", "border": "#CC5599"},
    "woody":     {"bg": "#8FAF70", "border": "#4A6B30"},
    "spicy":     {"bg": "#FF8C66", "border": "#CC3300"},
    "fresh":     {"bg": "#99E6E6", "border": "#006B8F"},
    "green":     {"bg": "#AADD77", "border": "#447722"},
    "powdery":   {"bg": "#DDAADD", "border": "#884488"},
    "fruity":    {"bg": "#FF9977", "border": "#CC4400"},
    "aldehydic": {"bg": "#D4C88A", "border": "#8A7B2A"},
    "other":     {"bg": "#CCCCCC", "border": "#888888"},
}


def infer_odor_family(name: str, smiles: str = "", logp: float = 0.0) -> str:
    n = name.lower()
    for family, keywords in ODOR_FAMILY_KEYWORDS.items():
        if any(kw in n for kw in keywords):
            return family

    if RDKIT_AVAILABLE and smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            lactone_pat  = Chem.MolFromSmarts("[O;R][C;R](=O)")
            ester_pat    = Chem.MolFromSmarts("[CX3](=O)[OX2H0][#6]")
            aldehyde_pat = Chem.MolFromSmarts("[CX3H1](=O)[#6]")
            alcohol_pat  = Chem.MolFromSmarts("[OX2H][#6]")
            if mol.HasSubstructMatch(lactone_pat):
                return "vanilla"
            if mol.HasSubstructMatch(ester_pat):
                return "fruity"
            if mol.HasSubstructMatch(aldehyde_pat):
                return "aldehydic"
            ring_count = mol.GetRingInfo().NumRings()
            if ring_count >= 2 and logp >= 3.5:
                return "musk"
            if mol.HasSubstructMatch(alcohol_pat):
                return "floral"

    return "other"


@st.cache_data(show_spinner=False)
def build_odor_df(_df: pd.DataFrame) -> pd.DataFrame:
    out = _df.copy()
    out["Odor_Family"] = out.apply(
        lambda r: infer_odor_family(r["Name"], str(r.get("SMILES", "")), float(r.get("LogP", 0.0))),
        axis=1,
    )
    return out


def build_odor_network_html(network_df: pd.DataFrame, selected_families: List[str], max_nodes: int) -> str:
    subset = network_df[network_df["Odor_Family"].isin(selected_families)]
    # Prioritise musk & vanilla, then others
    priority = subset[subset["Odor_Family"].isin(["musk", "vanilla"])]
    rest = subset[~subset["Odor_Family"].isin(["musk", "vanilla"])]
    combined = pd.concat([priority, rest]).head(max_nodes).reset_index(drop=True)

    nodes: List[dict] = []
    family_to_ids: Dict[str, List[int]] = {}

    for i, row in combined.iterrows():
        fam = row["Odor_Family"]
        col = FAMILY_COLORS.get(fam, FAMILY_COLORS["other"])
        highlight = fam in ("musk", "vanilla")
        name_label = row["Name"][:22] + ("…" if len(row["Name"]) > 22 else "")
        limit_str = f"{row['Limit_Cat4']}%" if str(row.get("Limit_Cat4", "")) not in ("", "nan") else "—"
        nodes.append({
            "id": i,
            "label": name_label,
            "title": f"<b>{row['Name']}</b><br>Family: {fam}<br>Status: {row.get('Status','')}<br>Limit: {limit_str}",
            "color": {
                "background": col["bg"],
                "border": col["border"],
                "highlight": {"background": "#FFFF99", "border": "#FF6600"},
            },
            "group": fam,
            "size": 22 if highlight else 14,
            "font": {"size": 13 if highlight else 10, "color": "#222222"},
            "borderWidth": 3 if highlight else 1,
        })
        family_to_ids.setdefault(fam, []).append(i)

    edges: List[dict] = []
    eid = 0
    for fam, ids in family_to_ids.items():
        col = FAMILY_COLORS.get(fam, FAMILY_COLORS["other"])
        limited = ids[:12]  # cap edges per family to avoid overload
        for a in range(len(limited)):
            for b in range(a + 1, len(limited)):
                edges.append({
                    "id": eid,
                    "from": limited[a],
                    "to": limited[b],
                    "color": {"color": col["border"], "opacity": 0.25},
                    "width": 1,
                })
                eid += 1

    legend_html = ""
    for fam in sorted(FAMILY_COLORS.keys()):
        if fam == "other":
            continue
        count = len(family_to_ids.get(fam, []))
        if count == 0:
            continue
        col = FAMILY_COLORS[fam]
        bold = "font-weight:700;" if fam in ("musk", "vanilla") else ""
        legend_html += (
            f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0;{bold}">'
            f'<div style="width:13px;height:13px;border-radius:50%;background:{col["bg"]};'
            f'border:2px solid {col["border"]};flex-shrink:0;"></div>'
            f'<span style="font-size:12px;color:#ddd;">{fam} ({count})</span></div>'
        )

    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)

    return """
<!DOCTYPE html>
<html>
<head>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body { margin: 0; background: #12122a; }
    #net { width: 100%; height: 560px; background: #12122a; border: 1px solid #333; }
    #legend { position:absolute; top:10px; right:12px; background:rgba(18,18,42,0.92);
              padding:10px 14px; border-radius:8px; border:1px solid #444;
              max-height:540px; overflow-y:auto; }
    #legend h4 { margin:0 0 7px; font-size:13px; color:#aaa;
                 border-bottom:1px solid #555; padding-bottom:4px; font-family:Georgia,serif; }
  </style>
</head>
<body>
  <div style="position:relative;">
    <div id="net"></div>
    <div id="legend">
      <h4>Odor Families</h4>
      LEGEND_HTML
    </div>
  </div>
  <script>
    var nodes = new vis.DataSet(NODES_JSON);
    var edges = new vis.DataSet(EDGES_JSON);
    var net = new vis.Network(
      document.getElementById("net"),
      { nodes: nodes, edges: edges },
      {
        physics: {
          solver: "forceAtlas2Based",
          forceAtlas2Based: {
            gravitationalConstant: -55,
            centralGravity: 0.004,
            springLength: 130,
            springConstant: 0.09,
            damping: 0.45,
            avoidOverlap: 0.6
          },
          stabilization: { iterations: 220 }
        },
        nodes: { shape: "dot", shadow: { enabled: true, size: 7, x: 2, y: 2 } },
        edges: { smooth: { enabled: true, type: "continuous" } },
        interaction: { hover: true, tooltipDelay: 150, hideEdgesOnDrag: true, navigationButtons: true }
      }
    );
  </script>
</body>
</html>
""".replace("NODES_JSON", nodes_json).replace("EDGES_JSON", edges_json).replace("LEGEND_HTML", legend_html)


def render_structure(smiles: str, label: str) -> None:
    """Render a 2D structure from SMILES with RDKit or a network fallback."""
    if not smiles:
        st.info("No SMILES available for structure rendering.")
        return

    if RDKIT_AVAILABLE:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.warning("SMILES could not be parsed for 2D rendering.")
            return
        img = Draw.MolToImage(mol, size=(320, 320))
        st.image(img, caption=f"2D structure of {label}")
        return

    # Browser-side rendering fallback using SmilesDrawer.
    # This avoids server-side RDKit and often works even when API image fallback fails.
    smiles_js = smiles.replace("\\", "\\\\").replace("`", "\\`")
    canvas_id = f"mol_canvas_{abs(hash(smiles_js)) % 1_000_000_000}"
    html = f"""
    <div style="display:flex;flex-direction:column;gap:6px;">
      <canvas id="{canvas_id}" width="320" height="320" style="border:1px solid #dfe3e8;border-radius:8px;background:#fff;"></canvas>
      <div style="font-size:12px;color:#4b5563;">2D structure of {label}</div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/smiles-drawer@2.1.7/dist/smiles-drawer.min.js"></script>
    <script>
      (function() {{
        const smiles = `{smiles_js}`;
        const target = document.getElementById("{canvas_id}");
        if (!window.SmilesDrawer || !target) return;
        const drawer = new SmilesDrawer.Drawer({{
          width: 320,
          height: 320,
          bondThickness: 1.2,
          compactDrawing: true
        }});
        SmilesDrawer.parse(smiles, function(tree) {{
          drawer.draw(tree, target, "light", false);
        }});
      }})();
    </script>
    """
    components.html(html, height=360)

    # Additional network-image fallback for environments where JS CDN is blocked.
    encoded = quote(smiles, safe="")
    for url in [
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded}/PNG",
        f"https://cactus.nci.nih.gov/chemical/structure/{encoded}/image",
    ]:
        try:
            with urlopen(url, timeout=8) as response:
                image_bytes = response.read()
                if image_bytes:
                    st.image(image_bytes)
                    return
        except Exception:
            continue


df = load_directory_data()
lookup = build_limit_lookup()

if df.empty:
    st.error("No data available. Add IFRA and watchlist CSV files to run this app.")
    st.stop()


tab0, tab1, tab2, tab3, tab4 = st.tabs(
    ["Directory", "Consumer View", "Scientist View", "Regulatory Audit", "Odor Network"]
)

with tab0:
    st.subheader("Molecule Intelligence Directory")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total", len(df))
    with c2:
        st.metric("Banned", int((df["Category"] == "Banned").sum()))
    with c3:
        st.metric("Restricted", int((df["Category"] == "Restricted").sum()))

    search_text = st.text_input("Search by ingredient name", value="").strip().lower()
    filter_opt = st.radio("Filter", ["All", "Banned", "Restricted", "Safe"], horizontal=True)
    sort_opt = st.selectbox(
        "Sort",
        ["Alphabetical", "Year Banned (Newest)", "Year Banned (Oldest)"],
    )

    filtered_df = df.copy()
    if search_text:
        filtered_df = filtered_df[filtered_df["Name"].str.lower().str.contains(search_text, na=False)]

    if filter_opt != "All":
        filtered_df = filtered_df[filtered_df["Category"] == filter_opt]

    if sort_opt == "Alphabetical":
        filtered_df = filtered_df.sort_values("Name", ascending=True)
    else:
        year_num = pd.to_numeric(filtered_df["Year"], errors="coerce")
        filtered_df = filtered_df.assign(Year_Num=year_num)
        filtered_df = filtered_df.sort_values(
            "Year_Num", ascending=(sort_opt == "Year Banned (Oldest)")
        ).drop(columns=["Year_Num"])

    st.write(f"Showing {len(filtered_df)} result(s).")
    st.dataframe(
        filtered_df[["Name", "Category", "Year", "Limit_Cat4", "Safety_Risk"]],
        use_container_width=True,
        height=500,
    )

    st.download_button(
        "Download current view as CSV",
        filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="directory_view.csv",
        mime="text/csv",
    )

with tab1:
    st.subheader("Consumer ingredient view")
    ingredient = st.selectbox("Select ingredient", sorted(df["Name"].tolist()))
    row = df[df["Name"] == ingredient].iloc[0]

    st.markdown(f"### {row['Name']}")
    status = row["Status"]

    if status == "Banned":
        st.error("BANNED in perfumes.")
    elif status == "Restricted":
        st.warning("RESTRICTED. Safe only within concentration limits.")
    else:
        st.success("Safe / currently unregulated.")

    st.write(f"**Safety context:** {row['Safety_Risk']}")
    st.write(f"**Category 4 limit:** {row['Limit_Cat4']}%")
    st.write(f"**Rule year:** {row['Year']}")

with tab2:
    st.subheader("Scientist view")
    target_name = st.selectbox("Select target molecule", sorted(df["Name"].tolist()), key="target")
    target_row = df[df["Name"] == target_name].iloc[0]
    target_smiles = str(target_row.get("SMILES", ""))

    st.write(f"**SMILES:** `{target_smiles}`")
    st.write(f"**LogP:** {target_row.get('LogP', 0.0)}")

    render_structure(target_smiles, target_name)

    st.markdown("#### Structural replacement candidates")
    safe_pool = df[df["Category"] == "Safe"]

    if st.button("Compute replacements"):
        with st.spinner("Computing Tanimoto similarity on Morgan fingerprints..."):
            repl_df = compute_replacements(target_smiles, safe_pool, top_k=5)

        if repl_df.empty:
            st.warning("No valid replacement candidates found for this target.")
        else:
            st.dataframe(repl_df, use_container_width=True)

with tab3:
    st.subheader("Formula compliance audit")
    st.write("Upload a CSV with columns: `Ingredient, Percentage`.")

    uploaded = st.file_uploader("Formula CSV", type=["csv"])
    use_sample = st.button("Use sample_formula.csv")

    formula_df = pd.DataFrame()
    if uploaded is not None:
        formula_df = parse_formula_file(uploaded)
    elif use_sample:
        formula_df = parse_formula_file(None)

    if not lookup:
        st.warning("IFRA source data is not available. Audit cannot run.")

    if not formula_df.empty and lookup:
        report_df = audit_formula(formula_df, lookup)

        fail_count = int((report_df["Status"] == "FAIL").sum())
        pass_count = int((report_df["Status"] == "PASS").sum())

        c1, c2 = st.columns(2)
        with c1:
            st.metric("PASS", pass_count)
        with c2:
            st.metric("FAIL", fail_count)

        if fail_count > 0:
            st.error("Formula is non-compliant for Category 4.")
        else:
            st.success("Formula is compliant for Category 4.")

        st.dataframe(report_df, use_container_width=True)
        st.download_button(
            "Download audit report",
            report_df.to_csv(index=False).encode("utf-8"),
            file_name="ifra_audit_report.csv",
            mime="text/csv",
        )
    elif uploaded is not None and formula_df.empty:
        st.error("CSV format invalid. Required columns: Ingredient, Percentage")

with tab4:
    st.subheader("Odor Similarity Network")
    st.caption(
        "Each node is a fragrance ingredient. Nodes of the same colour share an odour family "
        "('smell the same'). Edges link molecules within the same family. "
        "Musk and Vanilla nodes are larger and bolder."
    )

    odor_df = build_odor_df(df)

    all_families = sorted(odor_df["Odor_Family"].unique().tolist())
    default_families = [f for f in ["musk", "vanilla", "floral", "citrus", "woody"] if f in all_families]

    col_a, col_b = st.columns([3, 1])
    with col_a:
        selected = st.multiselect(
            "Show odour families",
            options=all_families,
            default=default_families,
        )
    with col_b:
        max_n = st.slider("Max molecules", min_value=30, max_value=300, value=120, step=10)

    if not selected:
        st.info("Select at least one odour family to display the network.")
    else:
        counts = odor_df[odor_df["Odor_Family"].isin(selected)]["Odor_Family"].value_counts()
        cols = st.columns(min(len(counts), 6))
        for i, (fam, cnt) in enumerate(counts.items()):
            with cols[i % len(cols)]:
                st.metric(fam, cnt)

        html_src = build_odor_network_html(odor_df, selected, max_n)
        components.html(html_src, height=600, scrolling=False)

        with st.expander("Raw family assignments"):
            show_cols = ["Name", "Odor_Family", "Status", "Limit_Cat4"]
            fam_view = odor_df[odor_df["Odor_Family"].isin(selected)][show_cols].sort_values("Odor_Family")
            st.dataframe(fam_view, use_container_width=True, height=300)
