# Nose What's Legal

### Smell Better, Legally.

Live app URL:

https://nose-what-s-legal.onrender.com/

## Overview
This repository builds an end-to-end cheminformatics workflow for fragrance regulatory analysis, focused on IFRA Category 4 (Fine Fragrance) use-cases.

The system connects:
- document-derived regulatory data,
- chemical structure normalization,
- molecular feature engineering,
- predictive risk modeling,
- and application-layer compliance tools.

It is designed to answer practical formulation questions:
- Is this ingredient banned, restricted, or currently unregulated?
- Which unregulated molecules are structurally high-risk?
- What safer structural alternatives are available?
- Is a formula compliant with IFRA Category 4 concentration limits?
- What a molecule could smell like (structure-based estimate with confidence)?

## What This Project Implements

### 1) Regulatory Data Pipeline
The pipeline extracts and normalizes IFRA ingredient-level records, then organizes the data into machine-usable tables.

Key outputs:
- ingredient identity fields (`ingredient_name`, `synonyms`, `cas_number`)
- regulatory context (`category_4_limit_percent`, `reason`, `rule_year`)

### 2) Chemical Identifier Resolution
The project resolves structures from identifier text using CAS and name-based lookups:
- CAS-first strategy
- name fallback strategy
- local cache to avoid repeated lookups (`smiles_cache.json`)

Primary output:
- `ifra_category4_smiles.csv`

### 3) Molecular Featurization
Resolved SMILES are transformed into ML-ready vectors:
- RDKit descriptors (`MolWt`, `LogP`, `TPSA`)
- Morgan fingerprints (ECFP-like 2048-bit representation)

Primary output:
- `ifra_category4_features.csv`

### 4) Modeling and Risk Analytics
The project includes:
- similarity search with Jaccard/Tanimoto over bit vectors,
- classification over molecular fingerprints for risk-reason patterns,
- watchlist generation for unregulated molecules with high structural proximity to restricted compounds.

### 5) Application Layer
Two interfaces are provided:
- `main.py`: FastAPI backend with searchable molecule and audit endpoints.
- `app.py`: Streamlit dashboard for directory browsing, chemist/regulatory views, and compliance demonstration.

#### Dashboard Features:
- **Directory Tab**: Browse all regulated and high-risk molecules with search, filter, and sort capabilities.
- **Consumer View**: Simple ingredient lookup with regulatory status, safety context, and concentration limits.
- **Scientist View**: Detailed molecular profiles with SMILES notation, LogP descriptors, and structure visualization (2D rendering via RDKit or SmilesDrawer).
- **Regulatory Audit**: Formula compliance checker that validates ingredient concentrations against IFRA Category 4 limits.
- **Odor Network**: Interactive force-directed graph visualizing fragrance ingredients grouped by odor families (musk, vanilla, floral, citrus, woody, spicy, fresh, green, powdery, fruity, aldehydic). Musk and vanilla nodes are highlighted for regulatory importance. Powered by vis-network library for dynamic exploration.

Recent product updates in the FastAPI + web UI flow include:
- structure-based odor inference (`odor_profile`, `odor_basis`, `odor_confidence`),
- confidence- and category-based odor filters,
- confidence-colored odor tag chips in cards/intel/sidebar,
- restricted ingredient grouping by B/C/D grades,
- resilient 3D handling with fallback behavior when coordinates are unavailable.

## Results

Run `python scripts/validation.py` to regenerate every figure below into
[`validation_report.md`](validation_report.md). Fixed seeds; reproducible from the raw CSVs.
Scaffold-split figures shift by a point or two across RDKit and scikit-learn versions
(scaffold perception and Random Forest tie-breaking both change); the random-split
figures and all of the conclusions are stable. The report records the environment it ran in.

**Pipeline**

- 484 Category 4 ingredient records extracted and normalised.
- 83.5% resolved to a machine-readable structure. The remaining 16.5% are naturals,
  essential oils and undefined botanical extracts, which have no single structure —
  every structural claim below is scoped to the resolved subset.
- 319 molecules carry both a parseable structure and a restriction-reason label.

**Restriction-reason classification — what the model can and cannot do**

| model | random split | scaffold split |
| --- | ---: | ---: |
| Majority class | 72.7% ± 0.7 | 70.5% ± 12.4 |
| Ingredient name, character n-grams | 79.4% ± 3.1 | 67.5% ± 12.9 |
| Descriptors only | 90.9% ± 3.1 | 64.5% ± 11.3 |
| Morgan fingerprints | **94.5% ± 2.9** | **71.3% ± 11.6** |

The 94.5% figure under a random split reproduces the number this project originally
reported. It does not survive contact with a proper evaluation:

- **45.5% of the labelled set is duplicate structures** — the same molecule under
  different names and registry numbers. A random split puts copies of one molecule on
  both sides of the test boundary.
- 96.9% of molecules share a Bemis-Murcko scaffold with another molecule, and a single
  scaffold accounts for 112 of 319.
- Under a scaffold-held-out split, **the model is indistinguishable from the
  majority-class baseline** (71.3% vs 70.5% — a 0.8 point difference against a ±11.6
  point fold spread), and indistinguishable from a model given only the ingredient's
  *name* and no chemistry at all (67.5%).
- Per-class recall on held-out scaffolds is 0.97 for the dominant class and 0.03, 0.00,
  0.00 for the other three. It is a majority-class predictor.

The honest summary: **on chemistry it has not seen, this model has no demonstrated
skill.** The classes are too imbalanced and the dataset too structurally redundant to
support the original claim.

**Watchlist — 69 candidates, 16 after deduplication, 1 externally corroborated**

- 19 of the 69 are structurally *identical* to an already-restricted molecule
  (median reported similarity across the list: 100%). 11 are duplicates of each other;
  7 match an IFRA name or synonym the CAS-based filter missed.
- **16 survive every deduplication check.** That is the real list.
- Of those, **amyl salicylate (CAS 2050-08-0)** appears among the substances added to the
  EU labelled-allergen list by Commission Regulation (EU) 2023/1545 — an independent
  corroboration the pipeline did not have access to.
- **cyclopentanone** is a demonstrable false positive: 6 heavy atoms, 10 fingerprint
  on-bits, scored 90.9% similar to a sixteen-membered macrocycle. Tanimoto over Morgan
  fingerprints is unreliable for very small molecules, and the 85% cut-off needs a
  minimum-size guard.

**Assumptions tested**

- Among genuinely similar but non-identical pairs (0.85 ≤ Tanimoto < 1.0), 84.8% share a
  restriction reason against a 56.8% baseline — real lift, but from only 33 such pairs,
  and 15.2% of them are activity cliffs.
- Pairwise structural similarity is essentially uncorrelated with the difference in
  permitted concentration (Spearman ρ = −0.09), so similarity does not tell a formulator
  what they actually need to know about a substitution.
- Molecular weight, LogP and TPSA alone separate IFRA-listed from TGSC-unlisted materials
  at ROC-AUC 0.885. Three bulk properties should not identify a regulatory decision; they
  are identifying dataset provenance. Anything trained to separate those two pools risks
  learning which spreadsheet a molecule came from.
- "Unregulated" is not "safe". This is a positive-unlabeled problem — one labelled class
  and an unlabeled pool containing an unknown number of undiscovered positives — and
  reported precision against that pool is not meaningful.

## Repository Layout
- `app.py`: Streamlit interface
- `streamlit_app.py`: Streamlit Community Cloud entrypoint
- `main.py`: FastAPI service
- `new_UI.html`: FastAPI-served frontend entry
- `public/index.html`: legacy static entry
- `scripts/extract_ifra_category4.py`: IFRA extraction utility
- `scripts/fetch_smiles.py`: SMILES resolution utility
- `scripts/featurize_molecules.py`: RDKit feature generation
- `scripts/featurize_molecules_deepchem.py`: optional DeepChem fingerprint path
- `scripts/ml_model.py`: model training and candidate assessment
- `scripts/find_substitutes.py`: replacement search
- `scripts/formula_auditor.py`: formula compliance checker
- `scripts/early_warning_scanner.py`: sampled watchlist scanner
- `scripts/generate_watchlist_full.py`: full-scale watchlist generator
- `sample_formula.csv`: demo formula for audit testing
- `ifra_category4_*.csv`, `AI_Predictive_Watchlist.csv`: prepared data artifacts

## Setup

### Streamlit Hosting Setup (minimal)
This install path is optimized for Streamlit Community Cloud deployment.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Full Local Pipeline Setup
Use this for API + scripts + data engineering workflows.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-full.txt
```

## Run the Apps

### FastAPI
```bash
uvicorn main:app --reload
```

### Streamlit
```bash
streamlit run streamlit_app.py
```

## Run Core Pipeline Scripts

### Featurize existing SMILES data
```bash
python3 scripts/featurize_molecules.py \
  --input ifra_category4_smiles.csv \
  --output ifra_category4_features.csv
```

### Train/test model and assess one candidate
```bash
python3 scripts/ml_model.py \
  --db ifra_category4_features.csv \
  --test_name "Raspberry Ketone" \
  --test_smiles "O=C(CC1=CC=C(C=C1)O)C"
```

### Formula audit
```bash
python3 scripts/formula_auditor.py --formula sample_formula.csv
```

## API Endpoints (FastAPI)
- `GET /api/directory`: full molecule directory payload
- `GET /api/search?q=<prefix>`: prefix-based molecule search
- `GET /api/molecule/{name}`: molecule detail payload including optional computed properties
- `POST /api/audit`: batch compliance audit

Odor-related fields returned by directory/molecule endpoints:
- `odor_profile`
- `odor_basis`
- `odor_confidence`

## Current Scope and Limitations
- The workflow is centered on IFRA Category 4 analysis.
- Structure resolution quality is bounded by external resolver coverage and naming consistency.
- The deployed interfaces are functional prototypes intended for analysis workflows, not yet hardened production services.

## Public Deployment (Streamlit Community Cloud)
1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud and click **Create app**.
3. Select repository: `Rimcheb/fragrance-ai-compliance-system`.
4. Set branch: `main`.
5. Set main file path: `streamlit_app.py`.
6. Deploy.
7. In app settings, keep visibility as **Public**.

The app is cloud-ready by default:
- `requirements.txt` is deployment-focused and lightweight.
- `.streamlit/config.toml` contains server/theme defaults.
- The UI gracefully handles environments where RDKit is unavailable.

## Public Deployment (Render, FastAPI)
Use this when you want the FastAPI + `new_UI.html` experience publicly available.

Live app URL:
- https://nose-what-s-legal.onrender.com/

Build command:
```bash
pip install --upgrade pip && pip install -r requirements-full.txt
```

Start command:
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

You can deploy directly with the existing `render.yaml` blueprint.

## Next Engineering Priorities
- Improve structured extraction coverage and validation for additional IFRA classes/categories.
- Add stronger model validation suites and calibrated uncertainty outputs.
- Package repeatable training/evaluation commands into a single CLI entrypoint.

---

## Deployment & performance

The original deployment was a FastAPI service on Render's free tier. That tier
spins a service down after ~15 minutes of inactivity, so the first visitor after
a quiet period waited through a full cold boot (container start plus importing
RDKit, scikit-learn, pandas and NumPy). Two deployment modes now exist.

### Static build — recommended for the public demo

Every `/api/*` response the frontend uses is a pure function of two CSVs that
never change at runtime, so none of it needs a server:

```bash
python scripts/build_static.py     # writes ./static
```

This precomputes the directory, the odor network, every per-molecule record
(descriptors, Tanimoto nearest neighbour, odor inference) and all 3D coordinate
blocks, then emits a patched copy of the UI with a small `window.fetch` shim
that answers `/api/*` from `data.json`. `search` and `audit` run in the browser.

Output is roughly 110 KB of HTML plus 330 KB of JSON on first paint, with 3D
coordinates split into one lazily-fetched file per molecule (~670 KB total,
never all at once). Drop `static/` on GitHub Pages, Netlify or Vercel
(`vercel.json` is configured for it) and the page loads in about a second with
no server, no cold start and no monthly cost.

### FastAPI service

Still the reference implementation, and now considerably faster:

- `requirements-api.txt` holds the runtime set only. The previous
  `requirements-full.txt` also installed `streamlit`, `pdfplumber`,
  `beautifulsoup4`, `pubchempy`, `cirpy`, `tqdm` and `Pillow`, none of which
  `main.py` imports — they only inflated the image and the cold start. Use
  `requirements-full.txt` for the offline pipeline in `scripts/`.
- 3D embedding moved off `/api/molecule/{name}` into `/api/molecule3d/{name}`,
  which the UI fetches lazily after the card has rendered. The molecule card
  used to block on an ETKDG embed plus MMFF optimisation.
- Descriptors, Morgan fingerprints, nearest-neighbour lookups, 3D blocks and
  PubChem responses are all `lru_cache`d; the PubChem timeout dropped from 10s
  to 3s so a slow upstream degrades to "no 3D" instead of hanging the response.
- `GZipMiddleware` compresses responses (the UI HTML alone is ~107 KB raw).
- Frontend libraries are pinned and served from jsDelivr with `defer`, replacing
  an unpinned unpkg build and a 3Dmol copy hosted on a university server.

Note that free-tier spin-down is a platform property, not a code problem — the
changes above shorten the cold boot but cannot remove it. A paid Render instance
or a host without sleep (e.g. Hugging Face Spaces) is the fix if the live API
must stay warm.
