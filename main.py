from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from typing import List
from urllib.parse import quote
from urllib.request import urlopen
from pathlib import Path
from functools import lru_cache
import pandas as pd

# Try importing RDKit for 3D coordinate generation
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

app = FastAPI(title="Nose What's Legal API")

BASE_DIR = Path(__file__).resolve().parent
IFRA_CSV_PATH = BASE_DIR / "ifra_category4_smiles.csv"
WATCHLIST_CSV_PATH = BASE_DIR / "AI_Predictive_Watchlist.csv"
UI_HTML_PATH = BASE_DIR / "new_UI.html"
PUBLIC_DIR_PATH = BASE_DIR / "public"

# Compress responses (the UI HTML is ~107 KB uncompressed)
app.add_middleware(GZipMiddleware, minimum_size=500)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load real databases
MOCK_DB = {}

# 1. Load IFRA Category 4 Restricted Items
try:
    ifra_df = pd.read_csv(IFRA_CSV_PATH)
    for _, row in ifra_df.iterrows():
        name = str(row["ingredient_name"])
        if pd.isna(name) or name.lower() == "nan":
            continue
            
        limit = row["category_4_limit_percent"]
        limit_val = float(limit) if pd.notna(limit) else 0.0
        
        # Decide status based on limit
        status = "Banned" if limit_val == 0.0 else "Restricted"
        
        reason_val = row.get("reason", "Regulatory Risk")
        if pd.isna(reason_val) or str(reason_val).lower() == "nan":
            risk = "Safety Risk (Unspecified)"
        else:
            risk = str(reason_val)
            
        rule_year = row.get("rule_year")
        year_str = "Unknown Year" if pd.isna(rule_year) else str(int(rule_year))

        smiles_val = str(row["smiles"]) if pd.notna(row["smiles"]) else ""

        MOCK_DB[name.lower()] = {
            "name": name,
            "smiles": smiles_val,
            "status": status,
            "risk": risk,
            "limit": limit_val,
            "year": year_str,
            "replacement": "Compute via KNN",
            "desc": "Official IFRA Regulated Material. CAS: " + str(row.get("cas_number", "Unknown")).split(";")[0]
        }
except Exception as e:
    print(f"Failed to load IFRA data: {e}")

# 2. Load AI Predictive Watchlist (Unregulated but High Risk)
try:
    watchlist_df = pd.read_csv(WATCHLIST_CSV_PATH)
    for _, row in watchlist_df.iterrows():
        name = str(row["Candidate_Name"])
        if pd.isna(name) or name.lower() == "nan":
            continue
            
        if name.lower() in MOCK_DB:
            continue
            
        twin = str(row["Restricted_Twin_Molecule"])
        smiles_val = str(row["Candidate_SMILES"]) if pd.notna(row["Candidate_SMILES"]) else ""
        
        MOCK_DB[name.lower()] = {
            "name": name,
            "smiles": smiles_val,
            "status": "Safe / Unregulated (High Risk Watchlist)",
            "risk": str(row.get("AI_Predicted_Risk", "Potential Risk")),
            "limit": 100.0,
            "year": "N/A",
            "replacement": f"Structural Twin: {twin} ({row['Structural_Similarity_Score']}%)",
            "desc": "Currently unregulated but flagged by AI due to high structural similarity to restricted materials."
        }
except Exception as e:
    print(f"Failed to load Watchlist data: {e}")

print(f"Loaded ingredient records: {len(MOCK_DB)}")

ODOR_FAMILY_KEYWORDS_NET = {
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


def infer_odor_family_net(name: str, smiles: str = "", logp: float = 0.0) -> str:
    n = name.lower()
    for family, keywords in ODOR_FAMILY_KEYWORDS_NET.items():
        if any(kw in n for kw in keywords):
            return family

    if RDKIT_AVAILABLE and smiles:
        try:
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
                if mol.GetRingInfo().NumRings() >= 2 and logp >= 3.5:
                    return "musk"
                if mol.HasSubstructMatch(alcohol_pat):
                    return "floral"
        except Exception:
            pass

    return "other"


@app.get("/api/network")
def get_network():
    nodes = []
    for i, (_, entry) in enumerate(MOCK_DB.items()):
        family = infer_odor_family_net(
            entry.get("name", ""),
            entry.get("smiles", ""),
            0.0,
        )
        nodes.append({
            "id": i,
            "name": entry["name"],
            "odor_family": family,
            "status": entry["status"],
            "limit": entry["limit"],
            "smiles": entry.get("smiles", ""),
        })
    return nodes


@app.get("/api/directory")
def get_directory():
    mols = []
    for entry in sorted(MOCK_DB.values(), key=lambda x: x["name"]):
        item = entry.copy()
        logp, mol_wt = compute_descriptors(item.get("smiles", ""))
        item["logp"] = logp
        item["mol_wt"] = mol_wt
        item.update(
            infer_odor_profile(
                item.get("name", ""),
                item.get("smiles", ""),
                logp if isinstance(logp, (int, float)) else None,
            )
        )
        mols.append(item)
    return mols


@app.get("/api/health")
def get_health():
    return {
        "loaded_records": len(MOCK_DB),
        "ifra_csv_exists": IFRA_CSV_PATH.exists(),
        "watchlist_csv_exists": WATCHLIST_CSV_PATH.exists(),
        "base_dir": str(BASE_DIR),
    }

@app.get("/")
def serve_home():
    return FileResponse(UI_HTML_PATH)

# Serve static files built for the frontend
app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR_PATH)), name="public")


@lru_cache(maxsize=2048)
def fetch_pubchem_molblock(smiles: str):
    """Fetch a 3D SDF block from PubChem as a fallback when RDKit is unavailable.

    Cached: the same molecule is never fetched twice for the life of the process.
    Timeout is short on purpose — this sits on a request path, and a slow
    PubChem should degrade to "no 3D" rather than hang the whole response.
    """
    if not smiles:
        return None

    try:
        encoded = quote(smiles, safe="")
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded}/SDF?record_type=3d"
        with urlopen(url, timeout=3) as response:
            text = response.read().decode("utf-8", errors="ignore")
            if text and "M  END" in text:
                return text
    except Exception as e:
        print("PubChem 3D fallback failed:", e)

    return None


@lru_cache(maxsize=4096)
def compute_descriptors(smiles: str):
    """LogP and molecular weight from the 2D structure. No 3D embedding needed."""
    if not (RDKIT_AVAILABLE and smiles):
        return ("N/A", "N/A")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ("N/A", "N/A")
        return (round(Descriptors.MolLogP(mol), 2), round(Descriptors.MolWt(mol), 2))
    except Exception:
        return ("N/A", "N/A")


@lru_cache(maxsize=4096)
def morgan_fp(smiles: str):
    if not (RDKIT_AVAILABLE and smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    except Exception:
        return None


@lru_cache(maxsize=4096)
def compute_replacement(smiles: str, current_key: str):
    """Nearest structural neighbour by Tanimoto over the candidate pool.

    NOTE: structural similarity is not toxicological equivalence. This is a
    shortlist for a human to check, not a safety verdict.
    """
    target_fp = morgan_fp(smiles)
    if target_fp is None:
        return "No coordinates for Tanimoto."

    from rdkit import DataStructs

    best_match, best_score = None, 0.0
    for k, v in MOCK_DB.items():
        if k == current_key:
            continue
        if not ("Unregulated" in v["status"] or v["status"] == "Restricted"):
            continue
        cand_fp = morgan_fp(v.get("smiles", ""))
        if cand_fp is None:
            continue
        score = DataStructs.TanimotoSimilarity(target_fp, cand_fp)
        if score > best_score:
            best_score, best_match = score, v["name"]

    if best_match:
        return f"{best_match} ({round(best_score * 100, 1)}% Tanimoto Match)"
    return "No known unregulated structural proxy found."


@lru_cache(maxsize=2048)
def compute_molblock(smiles: str):
    """3D coordinates for the viewer. Slow (ETKDG embed + MMFF), so it is
    cached and lives behind its own endpoint rather than blocking /api/molecule.
    """
    if not smiles:
        return None
    if RDKIT_AVAILABLE:
        try:
            mol2d = Chem.MolFromSmiles(smiles)
            if mol2d is None:
                raise ValueError("Invalid SMILES string")
            m = Chem.AddHs(mol2d)
            if AllChem.EmbedMolecule(m, AllChem.ETKDGv3()) == -1:
                if AllChem.EmbedMolecule(m, randomSeed=42) == -1:
                    raise ValueError("3D embedding failed — no coordinates generated")
            AllChem.MMFFOptimizeMolecule(m)
            return Chem.MolToMolBlock(m)
        except Exception as e:
            print("RDKit 3D error:", e)
    return fetch_pubchem_molblock(smiles)


def infer_odor_profile(name: str, smiles: str, logp_value=None):
    """Infer likely odor families from structure using transparent heuristics.

    This is a structure-odor estimate, not a definitive sensory claim.
    """
    name_l = (name or "").lower()
    smiles_s = (smiles or "").strip()

    name_hints = [
        ("vanillin", "vanilla, creamy, sweet", "name match: vanillin-like aromatic aldehyde", "high"),
        ("musk", "musky, powdery", "name match: musk class ingredient", "high"),
        ("limonene", "citrus, zesty", "name match: limonene terpene", "high"),
        ("linalool", "floral, citrus, fresh", "name match: linalool terpene alcohol", "high"),
        ("citral", "lemon, aldehydic", "name match: citral aldehyde terpene", "high"),
        ("coumarin", "sweet hay, tonka, almond", "name match: coumarin aromatic lactone", "high"),
    ]
    for key, profile, basis, confidence in name_hints:
        if key in name_l:
            return {"odor_profile": profile, "odor_basis": basis, "odor_confidence": confidence}

    if RDKIT_AVAILABLE and smiles_s:
        try:
            mol = Chem.MolFromSmiles(smiles_s)
            if mol is not None:
                families = []
                basis = []

                patterns = [
                    ("[CX3](=O)[OX2H0][#6]", "fruity, sweet", "ester motif"),
                    ("[CX3H1](=O)[#6]", "aldehydic, citrus/waxy", "aldehyde motif"),
                    ("[#6][CX3](=O)[#6]", "woody, floral-fruity", "ketone motif"),
                    ("[OX2H][#6]", "fresh, floral", "alcohol motif"),
                    ("c[OH]", "phenolic, clove-like/smoky", "phenolic aromatic OH"),
                    ("[OD2]([#6])[#6]", "sweet, anisic", "ether motif"),
                    ("[O;R][C;R](=O)", "creamy, coconut/peachy", "lactone ring motif"),
                ]

                for smarts, profile, why in patterns:
                    patt = Chem.MolFromSmarts(smarts)
                    if patt is not None and mol.HasSubstructMatch(patt):
                        families.append(profile)
                        basis.append(why)

                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                if aromatic_atoms >= 6:
                    families.append("woody, balsamic")
                    basis.append("aromatic ring system")

                ring_count = mol.GetRingInfo().NumRings()
                if ring_count >= 3 and isinstance(logp_value, (int, float)) and logp_value >= 3:
                    families.append("ambery, musky")
                    basis.append("polycyclic + lipophilic profile")

                dedup_fam = list(dict.fromkeys(families))
                dedup_basis = list(dict.fromkeys(basis))
                if dedup_fam:
                    if len(dedup_fam) >= 3:
                        confidence = "high"
                    elif len(dedup_fam) == 2:
                        confidence = "medium"
                    else:
                        confidence = "low"
                    return {
                        "odor_profile": ", ".join(dedup_fam[:3]),
                        "odor_basis": "; ".join(dedup_basis[:3]),
                        "odor_confidence": confidence,
                    }
        except Exception:
            pass

    fallback_notes = []
    fallback_basis = []
    if "=O" in smiles_s:
        fallback_notes.append("aldehydic/ketonic")
        fallback_basis.append("carbonyl pattern in SMILES")
    if "c1" in smiles_s or "c2" in smiles_s:
        fallback_notes.append("woody/balsamic")
        fallback_basis.append("aromatic pattern in SMILES")
    if "O" in smiles_s:
        fallback_notes.append("sweet/floral")
        fallback_basis.append("oxygenated functionality")

    if fallback_notes:
        return {
            "odor_profile": ", ".join(list(dict.fromkeys(fallback_notes))[:3]),
            "odor_basis": "; ".join(list(dict.fromkeys(fallback_basis))[:3]),
            "odor_confidence": "low",
        }

    return {
        "odor_profile": "unknown",
        "odor_basis": "insufficient structural evidence",
        "odor_confidence": "low",
    }

@app.get("/api/search")
def search_molecules(q: str = ""):
    if not q:
        return []
    q_lower = q.lower()
    # Find active matches that start with the query letter(s)
    return [{"name": data["name"]} for key, data in MOCK_DB.items() if key.startswith(q_lower)]

@app.get("/api/molecule/{name}")
def get_molecule_data(name: str, include_3d: bool = False):
    """Fast path: everything except 3D coordinates.

    3D embedding used to run here and made this endpoint take seconds. It now
    lives at /api/molecule3d/{name}, which the UI fetches lazily after the card
    has already rendered. Pass include_3d=true to get the old blocking behaviour
    (used by the static site builder).
    """
    key = name.lower()
    if key not in MOCK_DB:
        raise HTTPException(status_code=404, detail="Molecule not found in database.")

    data = MOCK_DB[key].copy()
    smiles = data.get("smiles", "")

    logp, mol_wt = compute_descriptors(smiles)
    data["logp"] = logp
    data["mol_wt"] = mol_wt

    if data["replacement"] == "Compute via KNN":
        data["replacement"] = compute_replacement(smiles, key)

    data["mol_block"] = compute_molblock(smiles) if include_3d else None
    data["mol_block_url"] = f"/api/molecule3d/{quote(data['name'])}" if smiles else None

    data.update(
        infer_odor_profile(
            data.get("name", ""),
            smiles,
            data.get("logp") if isinstance(data.get("logp"), (int, float)) else None,
        )
    )

    return data


@app.get("/api/molecule3d/{name}")
def get_molecule_3d(name: str):
    """3D coordinates only. Slow the first time per molecule, cached after."""
    key = name.lower()
    if key not in MOCK_DB:
        raise HTTPException(status_code=404, detail="Molecule not found in database.")
    return {"mol_block": compute_molblock(MOCK_DB[key].get("smiles", ""))}


# --- Regulatory Auditor Endpoint ---
class FormulaItem(BaseModel):
    ingredient: str
    percentage: float

class Formula(BaseModel):
    items: List[FormulaItem]
    filename: str = "uploaded_formula.csv"

@app.post("/api/audit")
def audit_formula(formula: Formula):
    results = []
    failed_items = []
    
    for item in formula.items:
        name_key = item.ingredient.lower().strip()
        db_entry = MOCK_DB.get(name_key)
        
        if db_entry:
            limit = db_entry["limit"]
            risk = db_entry["risk"]
            
            if "Banned" in db_entry["status"] or limit == 0.0:
                results.append({"status": "FAIL", "name": db_entry["name"], "formula_pct": item.percentage, "limit": "BANNED", "risk": risk})
                failed_items.append(db_entry["name"])
            elif item.percentage > limit:
                results.append({"status": "FAIL", "name": db_entry["name"], "formula_pct": item.percentage, "limit": f"{limit}%", "risk": risk})
                failed_items.append(db_entry["name"])
            else:
                results.append({"status": "PASS", "name": db_entry["name"], "formula_pct": item.percentage, "limit": f"{limit}%", "risk": "Permissible"})
        else:
            # Not in restricted DB, considered safe/unregulated
            results.append({"status": "PASS", "name": item.ingredient, "formula_pct": item.percentage, "limit": "N/A", "risk": "Unregulated"})
            
    is_compliant = len(failed_items) == 0
    return {
        "filename": formula.filename,
        "results": results,
        "compliant": is_compliant,
        "failed_items": failed_items
    }
