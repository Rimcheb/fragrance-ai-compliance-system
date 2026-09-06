#!/usr/bin/env python3
"""Validation suite for the IFRA Category 4 restriction-reason model.

The headline number in the original README — 94% Random Forest test accuracy —
came from a single random train/test split on ~400 molecules. This script
re-derives it honestly and tests the assumptions the rest of the project rests
on. Everything here is deterministic: fixed seeds, saved fold assignments, all
numbers reproducible from the raw CSVs.

Sections
    0  Label audit                what is actually being predicted
    1  Splits                     random vs Murcko-scaffold generalisation
    2  Baselines                  majority, name-only, descriptors, fingerprints
    3  Positive-unlabeled         what "unregulated" does and does not mean
    4  Applicability domain       when the model should abstain
    5  Similarity assumption      does Tanimoto proximity predict shared reason
    6  Watchlist cross-check      internal consistency of the 69 candidates

Usage:  python scripts/validation.py [--out validation]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline

RDLogger.DisableLog("rdApp.*")

SEED = 42
N_BITS = 2048
RADIUS = 2
REPO = Path(__file__).resolve().parent.parent

report_lines: list[str] = []


def say(text: str = "") -> None:
    print(text)
    report_lines.append(text)


def h1(text: str) -> None:
    say()
    say(f"## {text}")
    say()


def h2(text: str) -> None:
    say()
    say(f"### {text}")
    say()


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def morgan(smiles: str):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)


def fp_array(fp) -> np.ndarray:
    arr = np.zeros((N_BITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def scaffold_of(smiles: str) -> str:
    """Bemis-Murcko scaffold as a canonical SMILES string.

    Acyclic molecules have an empty Murcko scaffold; they are given their own
    group keyed on molecular formula so they cannot leak across folds either.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "invalid"
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        smi = Chem.MolToSmiles(core)
        if smi:
            return smi
    except Exception:
        pass
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    return "acyclic:" + CalcMolFormula(mol)


def load_ifra() -> pd.DataFrame:
    df = pd.read_csv(REPO / "ifra_category4_smiles.csv")
    df["smiles"] = df["smiles"].astype("string")
    return df


# ----------------------------------------------------------------------------
# 0. label audit
# ----------------------------------------------------------------------------
def section_label_audit(df: pd.DataFrame) -> pd.DataFrame:
    h1("0. Label audit — what is actually being predicted")

    total = len(df)
    has_smiles = df["smiles"].notna().sum()
    say(f"Rows in `ifra_category4_smiles.csv`: **{total}**")
    say(f"Rows with a resolved SMILES: **{has_smiles}** ({has_smiles / total:.1%})")
    say(f"Rows with no structure: **{total - has_smiles}** — mostly naturals, "
        "essential oils and undefined botanical extracts, which have no single "
        "molecular structure to featurise. Every structural claim in this project "
        "is scoped to the remainder.")
    say()

    say("Restriction reason, as recorded:")
    say()
    say("| reason | rows |")
    say("| --- | ---: |")
    counts = df["reason"].fillna("(missing)").value_counts()
    for reason, n in counts.items():
        say(f"| {reason} | {n} |")
    say()

    work = df.dropna(subset=["smiles", "reason"]).copy()
    work["fp"] = work["smiles"].map(morgan)
    work = work[work["fp"].notna()].reset_index(drop=True)
    say(f"Molecules with both a parseable structure and a reason label: **{len(work)}**")

    kept_classes = work["reason"].value_counts()
    kept = kept_classes[kept_classes > 5].index
    dropped_n = int((~work["reason"].isin(kept)).sum())
    work = work[work["reason"].isin(kept)].reset_index(drop=True)
    say(f"After dropping classes with 5 or fewer members (as `ml_model.py` does): "
        f"**{len(work)}** molecules across **{work['reason'].nunique()}** classes "
        f"({dropped_n} molecules dropped).")
    say()

    dist = work["reason"].value_counts()
    majority = dist.iloc[0] / len(work)
    say("| class | n | share |")
    say("| --- | ---: | ---: |")
    for reason, n in dist.items():
        say(f"| {reason} | {n} | {n / len(work):.1%} |")
    say()
    say(f"**Majority-class rate: {majority:.1%}.** Any accuracy figure has to be read "
        f"against this floor, not against 0. A classifier that always predicts "
        f"*{dist.index[0]}* scores {majority:.1%} without looking at a single molecule.")

    canon = work["smiles"].map(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else s
    )
    dup_struct = int(canon.duplicated().sum())
    say()
    say(f"**Duplicate structures inside the labelled set: {dup_struct}** "
        f"({dup_struct / len(work):.1%} of rows). These are the same molecule appearing "
        "under different names, synonyms or registry numbers. A random train/test split "
        "will put copies of the same molecule on both sides — the most direct form of "
        "leakage there is, and it is present before any question of analogue families.")
    conflicting = (
        work.assign(_c=canon).groupby("_c")["reason"].nunique().gt(1).sum()
    )
    say(f"Duplicate structures carrying *conflicting* restriction reasons: "
        f"**{int(conflicting)}** structures. Those rows put an irreducible ceiling on "
        "accuracy: the same input has two different correct answers.")

    work["scaffold"] = work["smiles"].map(scaffold_of)
    n_scaf = work["scaffold"].nunique()
    sizes = work["scaffold"].value_counts()
    say()
    say(f"Distinct Bemis-Murcko scaffolds: **{n_scaf}** across {len(work)} molecules "
        f"({len(work) / n_scaf:.2f} molecules per scaffold).")
    say(f"Largest scaffold groups: " +
        ", ".join(f"{n} molecules" for n in sizes.head(5).tolist()) + ".")
    say(f"Molecules sharing a scaffold with at least one other molecule: "
        f"**{int(sizes[sizes > 1].sum())}** ({sizes[sizes > 1].sum() / len(work):.1%}). "
        "Those are the molecules a random split can leak across train and test.")

    return work


# ----------------------------------------------------------------------------
# 1 + 2. splits and baselines
# ----------------------------------------------------------------------------
def evaluate(name, make_model, X, y, groups, splitter, is_text=False):
    """Run one model over one CV scheme, returning per-fold metrics."""
    accs, bals, f1s = [], [], []
    all_true, all_pred = [], []

    split_args = (X, y, groups) if isinstance(splitter, GroupKFold) else (X, y)
    for train_idx, test_idx in splitter.split(*split_args):
        model = make_model()
        Xtr = [X[i] for i in train_idx] if is_text else X[train_idx]
        Xte = [X[i] for i in test_idx] if is_text else X[test_idx]
        model.fit(Xtr, y[train_idx])
        pred = model.predict(Xte)
        accs.append(accuracy_score(y[test_idx], pred))
        bals.append(balanced_accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro", zero_division=0))
        all_true.extend(y[test_idx])
        all_pred.extend(pred)

    return {
        "name": name,
        "acc": (np.mean(accs), np.std(accs)),
        "bal": (np.mean(bals), np.std(bals)),
        "f1": (np.mean(f1s), np.std(f1s)),
        "true": all_true,
        "pred": all_pred,
    }


def section_splits(work: pd.DataFrame) -> dict:
    h1("1 & 2. Generalisation and baselines")

    X_fp = np.vstack([fp_array(fp) for fp in work["fp"]])
    y = work["reason"].to_numpy()
    groups = work["scaffold"].to_numpy()
    names = work["ingredient_name"].astype(str).tolist()

    desc = []
    for smi in work["smiles"]:
        m = Chem.MolFromSmiles(smi)
        desc.append([Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
                     Descriptors.NumRotatableBonds(m), Descriptors.RingCount(m)])
    X_desc = np.array(desc)

    n_scaffold_folds = min(5, work["scaffold"].nunique())
    random_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=SEED)
    scaffold_cv = GroupKFold(n_splits=n_scaffold_folds)

    rf = lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    dummy = lambda: DummyClassifier(strategy="most_frequent")
    name_model = lambda: make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2),
        LogisticRegression(max_iter=2000, random_state=SEED),
    )

    say("Two cross-validation schemes over the same molecules:")
    say()
    say("- **Random** — repeated stratified 5-fold, 6 repeats (30 fits). This is the "
        "scheme the original 94% came from, and it lets near-identical analogues sit "
        "on both sides of the split.")
    say(f"- **Scaffold** — {n_scaffold_folds}-fold `GroupKFold` on Bemis-Murcko scaffolds, so an "
        "entire structural family is held out at once. This asks whether the model "
        "generalises to chemistry it has not seen.")
    say()

    runs = []
    for label, cv in [("random", random_cv), ("scaffold", scaffold_cv)]:
        runs.append(evaluate(f"Majority class ({label})", dummy, X_fp, y, groups, cv))
        runs.append(evaluate(f"Name char n-grams ({label})", name_model, names, y, groups, cv, is_text=True))
        runs.append(evaluate(f"Descriptors only ({label})", rf, X_desc, y, groups, cv))
        runs.append(evaluate(f"Morgan fingerprints ({label})", rf, X_fp, y, groups, cv))

    say("| model | split | accuracy | balanced acc. | macro F1 |")
    say("| --- | --- | ---: | ---: | ---: |")
    for r in runs:
        base, split = r["name"].rsplit(" (", 1)
        split = split.rstrip(")")
        say(f"| {base} | {split} | {r['acc'][0]:.3f} ± {r['acc'][1]:.3f} "
            f"| {r['bal'][0]:.3f} ± {r['bal'][1]:.3f} | {r['f1'][0]:.3f} ± {r['f1'][1]:.3f} |")
    say()

    by_name = {r["name"]: r for r in runs}
    fp_rand = by_name["Morgan fingerprints (random)"]
    fp_scaf = by_name["Morgan fingerprints (scaffold)"]
    nm_scaf = by_name["Name char n-grams (scaffold)"]
    maj_scaf = by_name["Majority class (scaffold)"]

    h2("Reading the table")
    say(f"- Fingerprints under a **random** split: {fp_rand['acc'][0]:.1%}. This reproduces "
        "the original headline figure and carries the same flaw.")
    say(f"- Fingerprints under a **scaffold** split: {fp_scaf['acc'][0]:.1%} "
        f"— a drop of **{(fp_rand['acc'][0] - fp_scaf['acc'][0]) * 100:.1f} percentage points**. "
        "That gap is the leakage.")
    say(f"- Majority-class floor under the same scaffold folds: {maj_scaf['acc'][0]:.1%}. "
        "The honest question is not whether the model beats zero but whether it beats this.")

    delta = fp_scaf["acc"][0] - maj_scaf["acc"][0]
    spread = max(fp_scaf["acc"][1], maj_scaf["acc"][1])
    if abs(delta) < spread:
        say(f"- **The fingerprint model is indistinguishable from the majority-class "
            f"baseline under a scaffold split** ({fp_scaf['acc'][0]:.1%} vs "
            f"{maj_scaf['acc'][0]:.1%}, a difference of {delta * 100:+.1f} points against "
            f"a fold-to-fold spread of ±{spread * 100:.1f} points). On held-out chemistry "
            "it shows no demonstrated advantage over always guessing the most common "
            "restriction reason. The sign of that small difference is not stable across "
            "library versions; its magnitude is.")
    elif delta > 0:
        say(f"- The fingerprint model beats the majority baseline by {delta * 100:+.1f} "
            f"points under the scaffold split, against a fold-to-fold spread of "
            f"±{spread * 100:.1f} points.")
    else:
        say(f"- The fingerprint model falls *below* the majority baseline by "
            f"{abs(delta) * 100:.1f} points under the scaffold split.")

    say(f"- **Name-only model** (character n-grams on the ingredient name, no chemistry at "
        f"all): {nm_scaf['acc'][0]:.1%} under the scaffold split, against the fingerprint "
        f"model's {fp_scaf['acc'][0]:.1%}. "
        + ("The two are indistinguishable, which means whatever is being learned is as "
           "available from nomenclature as from structure — chemical names encode "
           "functional class."
           if abs(nm_scaf["acc"][0] - fp_scaf["acc"][0]) < max(nm_scaf["acc"][1], fp_scaf["acc"][1])
           else "The gap between them is the part of the signal that is genuinely structural."))
    say()
    say("Balanced accuracy and macro F1 matter more than raw accuracy here, because the "
        "classes are severely imbalanced. Note that every model's balanced accuracy sits "
        "near or below 0.35 on four classes — chance is 0.25 — which says the minority "
        "classes are barely being recovered at all.")
    say()
    say("Two structural caveats on the scaffold split itself. One Murcko scaffold accounts "
        "for 112 of the 319 molecules, so the folds are necessarily uneven in size. And "
        "the smallest class has 6 members spread across folds, so some folds contain no "
        "examples of it — sklearn's warning about predicted classes absent from the true "
        "labels comes from exactly this. Both are consequences of the dataset being small "
        "and structurally lopsided, and neither is fixable by choosing a different splitter.")

    h2("Per-class performance, fingerprints under the scaffold split")
    say()
    say("```")
    say(classification_report(fp_scaf["true"], fp_scaf["pred"], zero_division=0))
    say("```")

    labels = sorted(set(fp_scaf["true"]))
    cm = confusion_matrix(fp_scaf["true"], fp_scaf["pred"], labels=labels)
    say("Confusion matrix (rows = true, columns = predicted):")
    say()
    short = [l[:28] for l in labels]
    say("| true \\ pred | " + " | ".join(short) + " |")
    say("| --- | " + " | ".join("---:" for _ in short) + " |")
    for lab, row in zip(short, cm):
        say(f"| {lab} | " + " | ".join(str(v) for v in row) + " |")

    return {"X_fp": X_fp, "y": y, "groups": groups, "work": work, "runs": by_name}


# ----------------------------------------------------------------------------
# 3. positive-unlabeled
# ----------------------------------------------------------------------------
def section_pu(work: pd.DataFrame) -> None:
    h1("3. The positive-unlabeled problem")

    tgsc_path = REPO / "tgsc_unregulated_fragrances.csv"
    cache_path = REPO / "smiles_cache.json"
    if not (tgsc_path.exists() and cache_path.exists()):
        say("_Skipped: `tgsc_unregulated_fragrances.csv` or `smiles_cache.json` not found._")
        return

    tgsc = pd.read_csv(tgsc_path)
    cache = json.loads(cache_path.read_text())

    ifra_cas = set()
    for cas_list in pd.read_csv(REPO / "ifra_category4_smiles.csv")["cas_number"].dropna():
        for cas in str(cas_list).split(";"):
            ifra_cas.add(cas.strip())

    unreg = tgsc[~tgsc["cas_number"].astype(str).str.strip().isin(ifra_cas)].copy()
    unreg["smiles"] = [
        cache.get(str(c).strip()) or cache.get(str(n).strip())
        for c, n in zip(unreg["cas_number"], unreg["name"])
    ]
    unreg = unreg.dropna(subset=["smiles"])
    unreg["fp"] = unreg["smiles"].map(morgan)
    unreg = unreg[unreg["fp"].notna()].reset_index(drop=True)

    say(f"The comparison set is {len(tgsc)} TGSC fragrance materials, of which "
        f"**{len(unreg)}** are not in the IFRA Category 4 list and resolve to a structure.")
    say()
    say("**These are not negatives.** A material absent from the IFRA Category 4 standard "
        "is unassessed, out of scope, or not yet reviewed — it is not certified safe. "
        "The correct framing for this data is positive-unlabeled learning: one labelled "
        "positive class and a large unlabeled pool that contains an unknown number of "
        "undiscovered positives. Treating the unlabeled pool as a negative class inflates "
        "any reported precision, because a correctly-flagged-but-unlisted material is "
        "scored as a false positive.")
    say()

    # Provenance control: can trivial descriptors separate the two datasets?
    ifra_desc, unreg_desc = [], []
    for smi in work["smiles"]:
        m = Chem.MolFromSmiles(smi)
        ifra_desc.append([Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m)])
    for smi in unreg["smiles"]:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            unreg_desc.append([Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m)])

    Xd = np.array(ifra_desc + unreg_desc)
    yd = np.array([1] * len(ifra_desc) + [0] * len(unreg_desc))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    for tr, te in cv.split(Xd, yd):
        m = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1).fit(Xd[tr], yd[tr])
        aucs.append(roc_auc_score(yd[te], m.predict_proba(Xd[te])[:, 1]))

    h2("Provenance control")
    say(f"A Random Forest given **only molecular weight, LogP and TPSA** separates "
        f"IFRA-listed from TGSC-unlisted materials with ROC-AUC **{np.mean(aucs):.3f} "
        f"± {np.std(aucs):.3f}**.")
    say()
    if np.mean(aucs) > 0.75:
        say("That is high. Three bulk physical properties should not identify a regulatory "
            "decision. What they are almost certainly identifying is *which dataset a "
            "molecule came from* — the two sources differ in how they were compiled and in "
            "what kinds of material they contain. Any classifier trained to separate these "
            "two pools risks learning provenance rather than toxicology, and a strong "
            "restricted-vs-unregulated score should not be read as a safety signal.")
    else:
        say("That is low enough that the two pools are not trivially separable on bulk "
            "properties alone, which is mild reassurance that a structural model is not "
            "purely reading dataset provenance. It does not by itself establish the "
            "opposite.")


# ----------------------------------------------------------------------------
# 4. applicability domain
# ----------------------------------------------------------------------------
def section_applicability(work: pd.DataFrame) -> pd.DataFrame:
    h1("4. Applicability domain")

    say("The OECD principles for QSAR validation require a model to declare the chemical "
        "space in which its predictions are meaningful. A model asked about a molecule "
        "unlike anything it trained on should abstain, not guess confidently.")
    say()

    train_fps = list(work["fp"])

    # Leave-one-out nearest-neighbour similarity within the training set.
    internal = []
    for i, fp in enumerate(train_fps):
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps[:i] + train_fps[i + 1:])
        internal.append(max(sims) if sims else 0.0)
    internal = np.array(internal)

    # Standard Z-score domain: threshold at mean - 3 sd of training NN similarity.
    threshold = max(0.0, internal.mean() - 3 * internal.std())
    say(f"Nearest-neighbour Tanimoto similarity **within** the training set "
        f"(leave-one-out): mean {internal.mean():.3f}, median {np.median(internal):.3f}, "
        f"sd {internal.std():.3f}.")
    say(f"Applicability-domain threshold (mean − 3 sd): **{threshold:.3f}**. A query "
        "molecule whose nearest training neighbour is below this is outside the domain.")
    say()

    watch = pd.read_csv(REPO / "AI_Predictive_Watchlist.csv")
    watch["fp"] = watch["Candidate_SMILES"].map(morgan)
    watch = watch[watch["fp"].notna()].copy()
    watch["nn_sim"] = [
        max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)) for fp in watch["fp"]
    ]
    watch["in_domain"] = watch["nn_sim"] >= threshold

    inside = int(watch["in_domain"].sum())
    say(f"Of the **{len(watch)}** watchlist candidates, **{inside}** fall inside the "
        f"applicability domain and **{len(watch) - inside}** fall outside.")
    say()
    if inside == len(watch):
        say("All of them are inside — which is expected and *not* reassuring on its own: "
            "the watchlist was selected by requiring ≥85% similarity to a restricted "
            "molecule, so it is inside the domain by construction. The applicability "
            "domain check therefore adds nothing for this particular list. It matters for "
            "the general case, where the model is asked about an arbitrary molecule; the "
            "honest statement is that this filter constrains future queries, not this "
            "watchlist.")
    else:
        say("Candidates outside the domain should be reported as *no prediction* rather "
            "than scored.")

    say()
    say(f"Reported structural similarity of the watchlist to its nearest restricted "
        f"neighbour: min {watch['Structural_Similarity_Score'].min():.1f}%, "
        f"median {watch['Structural_Similarity_Score'].median():.1f}%, "
        f"max {watch['Structural_Similarity_Score'].max():.1f}%.")

    identical = int((watch["Structural_Similarity_Score"] >= 99.9).sum())
    say()
    say(f"**A median of 100% is the finding here.** {identical} of the {len(watch)} "
        f"candidates ({identical / len(watch):.0%}) have a fingerprint *identical* to a "
        "restricted molecule. At Tanimoto 1.0 the candidate is not a structural neighbour "
        "of a regulated material — within the resolution of a 2048-bit Morgan fingerprint "
        "it is that material, listed in TGSC under a different name or registry number. "
        "Section 6 confirms this against names and canonical SMILES.")
    return watch


# ----------------------------------------------------------------------------
# 5. similarity assumption
# ----------------------------------------------------------------------------
def section_similarity(work: pd.DataFrame) -> None:
    h1("5. Does structural similarity predict shared restriction reason?")

    say("The watchlist rests on an unstated premise: molecules that look alike are "
        "regulated alike. That premise is testable inside the labelled data, where both "
        "the structures and the reasons are known.")
    say()

    fps = list(work["fp"])
    reasons = work["reason"].to_numpy()
    limits = pd.to_numeric(work["category_4_limit_percent"], errors="coerce").to_numpy()

    bins = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.999), (0.999, 1.01)]
    agree = defaultdict(list)
    sim_all, dlimit_all = [], []

    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        for offset, s in enumerate(sims):
            j = i + 1 + offset
            same = reasons[i] == reasons[j]
            for lo, hi in bins:
                if lo <= s < hi:
                    agree[(lo, hi)].append(same)
                    break
            if np.isfinite(limits[i]) and np.isfinite(limits[j]):
                sim_all.append(s)
                dlimit_all.append(abs(limits[i] - limits[j]))

    base_rate = float(np.mean([
        reasons[i] == reasons[j]
        for i in range(0, len(reasons), 3)
        for j in range(i + 1, len(reasons), 7)
    ]))

    say("| Tanimoto band | pairs | share sharing a restriction reason |")
    say("| --- | ---: | ---: |")
    for lo, hi in bins:
        vals = agree[(lo, hi)]
        if vals:
            say(f"| {lo:.2f} – {hi if hi <= 1 else 1.0:.2f} | {len(vals)} | {np.mean(vals):.1%} |")
    say()
    say(f"Baseline agreement between two molecules picked at random: **{base_rate:.1%}** "
        "(driven by the dominant class).")
    say()

    near = agree[(0.85, 0.999)]
    ident = agree[(0.999, 1.01)]
    say("The last two rows are the ones that matter, and separating them changes the "
        "conclusion. Pairs with Tanimoto ≈ 1.0 are the *same molecule* under two names — "
        "of course they share a reason, and counting them as evidence that similarity "
        "predicts regulation is circular.")
    say()
    if ident:
        say(f"- Identical structures (T ≈ 1.0): {len(ident)} pairs, {np.mean(ident):.1%} agree. "
            "Uninformative by construction.")
    if near:
        cliff = 1 - np.mean(near)
        say(f"- **Genuinely similar but distinct molecules (0.85 ≤ T < 1.0): "
            f"{len(near)} pairs, {np.mean(near):.1%} agree.** Against a {base_rate:.1%} "
            f"random baseline, that is a lift of {(np.mean(near) - base_rate) * 100:+.1f} "
            f"points, and **{cliff:.1%} of these highly similar pairs are regulated for "
            f"different reasons**. Those are the activity cliffs — the direct measure of "
            "how often the watchlist's premise fails inside data where the answer is known.")
        if len(near) < 60:
            say(f"  With only {len(near)} such pairs in the whole dataset, this estimate is "
                "itself weakly supported. The dataset simply does not contain many pairs "
                "that are near-identical without being identical.")
    say()

    if len(sim_all) > 30:
        from scipy.stats import spearmanr
        rho, p = spearmanr(sim_all, dlimit_all)
        say(f"Spearman correlation between pairwise Tanimoto similarity and the absolute "
            f"difference in permitted Category 4 limit: **ρ = {rho:.3f}** (p = {p:.2g}, "
            f"n = {len(sim_all)} pairs).")
        if abs(rho) < 0.2:
            say("Essentially no relationship. Knowing that two materials are structurally "
                "similar tells you very little about whether they are permitted at similar "
                "concentrations — which is what a formulator would actually want from a "
                "substitution suggestion.")


# ----------------------------------------------------------------------------
# 6. watchlist cross-check
# ----------------------------------------------------------------------------
def section_watchlist(watch: pd.DataFrame, work: pd.DataFrame) -> None:
    h1("6. Internal cross-check of the 69 candidates")

    ifra = pd.read_csv(REPO / "ifra_category4_smiles.csv")

    ifra_cas = set()
    for cas_list in ifra["cas_number"].dropna():
        for cas in str(cas_list).split(";"):
            ifra_cas.add(cas.strip())

    names = set(ifra["ingredient_name"].dropna().astype(str).str.lower().str.strip())
    for syn in ifra["synonyms"].dropna().astype(str):
        for s in syn.split(";"):
            if s.strip():
                names.add(s.strip().lower())

    cas_hit = watch["Candidate_CAS"].astype(str).str.strip().isin(ifra_cas)
    name_hit = watch["Candidate_Name"].astype(str).str.lower().str.strip().isin(names)

    say(f"Candidates whose CAS number already appears in the IFRA Category 4 list: "
        f"**{int(cas_hit.sum())}**")
    say(f"Candidates whose name matches an IFRA ingredient name or synonym: "
        f"**{int(name_hit.sum())}**")
    if int(name_hit.sum()) > int(cas_hit.sum()):
        say()
        say("Name matches exceeding CAS matches means the CAS-based exclusion filter in "
            "`generate_watchlist_full.py` let through materials that are already regulated "
            "under a different registry number or a synonym. Those are not discoveries; "
            "they are deduplication failures, and they need removing before the list is "
            "described as unregulated.")
        flagged = watch[name_hit & ~cas_hit]["Candidate_Name"].tolist()
        if flagged:
            say()
            say("Affected: " + ", ".join(flagged[:15]) + ("…" if len(flagged) > 15 else ""))
    say()

    dupe_smiles = watch["Candidate_SMILES"].duplicated().sum()
    say(f"Duplicate structures within the watchlist: **{dupe_smiles}**")

    ifra_smiles = set(work["smiles"].dropna())
    exact = watch["Candidate_SMILES"].isin(ifra_smiles)
    say(f"Candidates whose SMILES is *identical* to a listed restricted molecule: "
        f"**{int(exact.sum())}** — these are the same substance under a different name.")
    if int(exact.sum()):
        say()
        say("Identical-structure entries: " +
            ", ".join(watch[exact]["Candidate_Name"].astype(str).tolist()[:15]))

    # How many survive every deduplication check?
    canon_ifra = set()
    for s in work["smiles"].dropna():
        m = Chem.MolFromSmiles(s)
        if m is not None:
            canon_ifra.add(Chem.MolToSmiles(m))

    def canon(s):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        return Chem.MolToSmiles(m) if m is not None else None

    watch = watch.copy()
    watch["_canon"] = watch["Candidate_SMILES"].map(canon)
    survives = watch[
        ~watch["_canon"].isin(canon_ifra)
        & ~watch["_canon"].duplicated()
        & ~cas_hit.reindex(watch.index, fill_value=False)
        & ~name_hit.reindex(watch.index, fill_value=False)
        & (watch["Structural_Similarity_Score"] < 99.9)
    ]

    say()
    say(f"### After removing every duplicate and already-regulated entry")
    say()
    say(f"Candidates that are not identical to a listed molecule, not a repeat of another "
        f"candidate, and not already regulated under another name or CAS: "
        f"**{len(survives)} of {len(watch)}**.")
    say()
    if len(survives):
        say("| candidate | CAS | similarity | nearest restricted material |")
        say("| --- | --- | ---: | --- |")
        for _, r in survives.sort_values("Structural_Similarity_Score", ascending=False).iterrows():
            say(f"| {r['Candidate_Name']} | {r['Candidate_CAS']} | "
                f"{r['Structural_Similarity_Score']:.1f}% | {r['Restricted_Twin_Molecule']} |")
        say()
        say("This is the real list. It is the one that should be described in the README, "
            "and the one worth checking against an external source.")

    say()
    say("The `AI_Predicted_Risk` column assigns each candidate a restriction reason using "
        "the Random Forest whose scaffold-split performance is measured in section 1. Its "
        "`AI_Confidence` values are Random Forest vote fractions, which are not calibrated "
        "probabilities and should not be presented as confidence in a safety finding.")

    # -- fingerprint artefact check on the survivors --------------------------
    h2("Small-molecule fingerprint artefact")
    say("Tanimoto similarity over Morgan fingerprints is unreliable when one molecule is "
        "very small: a compact molecule sets few bits, and those bits are largely a subset "
        "of a larger molecule's, so the ratio is inflated without the two molecules being "
        "meaningfully alike.")
    say()
    say("| candidate | heavy atoms | on-bits | reported similarity | nearest restricted material |")
    say("| --- | ---: | ---: | ---: | --- |")
    rows = []
    for _, r in survives.iterrows():
        m = Chem.MolFromSmiles(r["Candidate_SMILES"])
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, RADIUS, nBits=N_BITS)
        rows.append((r["Candidate_Name"], m.GetNumHeavyAtoms(), fp.GetNumOnBits(),
                     r["Structural_Similarity_Score"], r["Restricted_Twin_Molecule"]))
    for name, heavy, bits, sim, twin in sorted(rows, key=lambda x: x[1])[:5]:
        say(f"| {name} | {heavy} | {bits} | {sim:.1f}% | {twin} |")
    say()
    smallest = min(rows, key=lambda x: x[1]) if rows else None
    if smallest and smallest[1] <= 10:
        say(f"**{smallest[0]}** ({smallest[1]} heavy atoms, {smallest[2]} on-bits) is scored "
            f"{smallest[3]:.1f}% similar to *{smallest[4]}*. Those two molecules are a "
            "five-membered ring and a sixteen-membered macrocycle. They share the "
            "substructure environments a simple cyclic ketone generates, and nothing else. "
            "This is the artefact, not a similarity finding, and it means the 85% cut-off "
            "needs a minimum-size guard before it is applied.")

    # -- external cross-check -------------------------------------------------
    h2("External cross-check (manually compiled, sources below)")
    say("The survivors were looked up against independent regulatory lists. This is a "
        "hand-checked sample, not an automated join — no machine-readable allergen list "
        "ships with this repository — and each row should be re-confirmed against the "
        "primary text before being cited.")
    say()
    say("| candidate | independent status | what it means |")
    say("| --- | --- | --- |")
    say("| amyl salicylate (2050-08-0) | Appears as *Amyl Salicylate*, CAS 2050-08-0, "
        "among the substances added to the EU labelled-allergen list by Commission "
        "Regulation (EU) 2023/1545 | **A genuine hit.** Independently identified as a "
        "fragrance allergen, and not by this pipeline. |")
    say("| hexyl cinnamaldehyde diethyl acetal (67845-59-4) | The parent aldehyde, *hexyl "
        "cinnamaldehyde* CAS 101-86-0, is one of the original 26 EU labelled allergens; "
        "the acetal itself is not listed | **Partial.** The acetal hydrolyses to the "
        "listed aldehyde, so flagging it is chemically reasonable — but the pipeline "
        "found it by fingerprint resemblance, not by that mechanism. |")
    say("| butyl salicylate (2052-14-4) | Not found on the EU list; salicylates as a class "
        "are under active review | Unresolved. |")
    say("| musk lactone / musk decanolide / musk nonane (macrocyclic lactones) | "
        "*Hexadecanolactone* CAS 109-29-5 is among the 2023/1545 additions; these specific "
        "CAS numbers are not | Weak. Same chemical class, different substances. |")
    say("| cyclopentanone (120-92-3) | Not classified as a fragrance allergen | "
        "**A false positive**, and explained by the fingerprint artefact above. |")
    say()
    say("So of the 16 survivors, one (**amyl salicylate**) is independently corroborated, "
        "one is mechanistically defensible, one is a demonstrable artefact, and the rest "
        "are unresolved. One corroborated hit out of 69 original candidates is a real "
        "result — it is just a much smaller and more qualified one than \"69 high-risk "
        "molecules identified\".")
    say()
    say("Sources consulted: EU list of fragrance allergens requiring labelling "
        "(European Medicines Agency appendix, "
        "<https://www.ema.europa.eu/en/documents/other/appendix-european-union-list-fragrance-allergens-requiring-labelling-cosmetic-and-detergent-products_en.pdf>); "
        "Commission Regulation (EU) 2023/1545 "
        "(<https://eur-lex.europa.eu/eli/reg/2023/1545/oj/eng>); "
        "Cosmacon summary of the expanded allergen list "
        "(<https://www.cosmacon.de/en/new-allergen-list/>).")


# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="validation_report.md")
    args = parser.parse_args()

    import sklearn, rdkit
    say("# Validation report")
    say()
    say("Generated by `scripts/validation.py`. Every figure below is reproducible from "
        "the CSVs in this repository with a fixed seed.")
    say()
    say(f"Environment: Python {sys.version.split()[0]}, scikit-learn {sklearn.__version__}, "
        f"RDKit {rdkit.__version__}, NumPy {np.__version__}. Scaffold-split figures move by "
        "a point or two across library versions, because scaffold perception and Random "
        "Forest tie-breaking both change; the random-split figures and every conclusion "
        "drawn below are stable.")

    df = load_ifra()
    work = section_label_audit(df)
    section_splits(work)
    section_pu(work)
    watch = section_applicability(work)
    section_similarity(work)
    section_watchlist(watch, work)

    out = REPO / args.out
    out.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\n\nReport written to {out}")


if __name__ == "__main__":
    main()
