"""
Lead optimization evaluation for SMDD-Bench Type-4 tasks (ADMET-only, no Boltz2).

Scoring in [0, 1]:
  - 0.0 : invalid SMILES, unchanged-from-reference molecule, or any RDKit hard constraint fails
  - 1.0 : all hard constraints pass

Note: ADMET objective evaluation (admet_ai) is omitted due to container dependency
conflicts, so this can't verify the agent's molecule actually IMPROVED on the
optimization objective vs. baseline (the real SMDD-Bench scoring does, via
assess_optimization_objectives). Hard constraint satisfaction is a necessary
(though not sufficient) condition for a valid lead optimization result. As a
partial stand-in for the missing objective check, a molecule canonically
identical to the reference is rejected outright -- it has zero improvement by
definition, which fails every objective threshold in this dataset (all > 0),
so without this guard echoing the reference back verbatim scored a free 1.0.

RDKit is installed on first use if missing.
"""

import importlib
import importlib.util
import os
import re
import subprocess
import sys


def _ensure_rdkit():
    if importlib.util.find_spec("rdkit") is None:
        print("[lead_opt_eval] rdkit not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rdkit", "--no-deps", "-q"])


def _check_hard_constraints(smiles: str, reference_smiles: str) -> bool:
    import os, sys
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, AllChem, DataStructs, RDConfig
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False
    if any(a.GetNumRadicalElectrons() > 0 for a in mol.GetAtoms()):
        return False
    if "." in smiles:
        return False

    if Descriptors.MolWt(mol) >= 600:
        return False
    logp = Crippen.MolLogP(mol)
    if not (-1 <= logp <= 5):
        return False
    if Descriptors.TPSA(mol) >= 140:
        return False
    if Lipinski.NumHDonors(mol) > 5:
        return False
    if Lipinski.NumHAcceptors(mol) > 10:
        return False
    if Lipinski.NumRotatableBonds(mol) > 10:
        return False
    if not (-2 <= Chem.GetFormalCharge(mol) <= 2):
        return False
    if sascorer.calculateScore(mol) >= 4.5:
        return False

    pains_params = FilterCatalogParams()
    pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    if FilterCatalog(pains_params).HasMatch(mol):
        return False

    brenk_params = FilterCatalogParams()
    brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    if FilterCatalog(brenk_params).HasMatch(mol):
        return False

    ref_mol = Chem.MolFromSmiles(reference_smiles)
    if ref_mol is None:
        return False

    # The reference molecule trivially satisfies every constraint above (it's
    # already a valid, drug-like starting point) and has Tanimoto-to-self =
    # 1.0, so echoing it back verbatim -- doing zero actual optimization --
    # passes all of them for free. The real SMDD-Bench scoring catches this
    # via assess_optimization_objectives (agent's ADMET-AI-predicted
    # properties must improve on the baseline by each objective's threshold,
    # and an unchanged molecule has improvement=0, which fails every
    # threshold in this dataset since they're all > 0); that check is
    # omitted here due to admet_ai's container dependency conflicts, so
    # reject an unchanged molecule directly as the closest available proxy.
    if Chem.MolToSmiles(mol) == Chem.MolToSmiles(ref_mol):
        return False

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_ref = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
    if DataStructs.TanimotoSimilarity(fp, fp_ref) < 0.7:
        return False

    return True


def _extract_smiles(output: str) -> str:
    """
    Extract the model's SMILES answer from raw output that may include reasoning before it.

    The prompt says "Output ONLY a single valid SMILES string, nothing else", but reasoning
    models overwhelmingly ignore that and prepend an explanation. Naively taking the first
    whitespace token then grabs the first English word of the explanation instead of the answer,
    scoring a hard 0 regardless of whether the real answer is a valid, well-optimized molecule.

    Three passes, most reliable first, each candidate validated with Chem.MolFromSmiles before
    being accepted:
      1. Whole-line match: a full line that parses as a molecule on its own. Strongest signal --
         a rationale sentence that just mentions a fragment can't parse as a standalone molecule,
         so it can't be confused with the real answer even if the model explains itself after
         stating it.
      2. Backtick-quoted substring: answer embedded inline within one line.
      3. Right-to-left token scan: last resort for an answer run together in prose with no line
         breaks -- scanned from the end since the answer is usually stated last.
    Falls back to the original first-token behavior if nothing parses, so a genuinely invalid
    response still correctly scores 0.
    """
    from rdkit import Chem

    def _valid(cand: str) -> bool:
        # RDKit's SMILES parser silently truncates at the first space and parses just the
        # prefix (.smi-file convention) -- without this guard, "I cannot provide..." would
        # "successfully" parse as a bogus one-atom molecule (I = iodine).
        if not cand or " " in cand:
            return False
        return Chem.MolFromSmiles(cand) is not None

    stripped = output.strip()
    if not stripped:
        return ""

    # Pass 1: whole-line match.
    for line in stripped.splitlines():
        cand = line.strip().strip('`"\'')
        if _valid(cand):
            return cand

    # Pass 2: backtick-quoted substring.
    for match in re.findall(r"`([^`]+)`", stripped):
        cand = match.strip()
        if _valid(cand):
            return cand

    # Pass 3: right-to-left token scan.
    tokens = stripped.split()
    trailing_punct = ".,;:!?\"'"
    for token in reversed(tokens):
        cand = token.strip('"\'')
        if _valid(cand):
            return cand
        cand2 = cand.rstrip(trailing_punct)
        if cand2 != cand and _valid(cand2):
            return cand2

    # Fallback: original first-token behavior -- a genuinely invalid response still scores 0.
    return tokens[0].strip('`"\'') if tokens else ""


def score_lead_opt(output: str, reference_smiles: str, baseline_values: dict,
                   objectives: list, hold_constant: list) -> float:
    """
    Score a lead optimization output on RDKit hard constraints only.

    Args:
        output:           Raw model output (SMILES extracted from it).
        reference_smiles: Reference ligand SMILES for Tanimoto check.
        baseline_values:  Unused (retained for API compatibility).
        objectives:       Unused (retained for API compatibility).
        hold_constant:    Unused (retained for API compatibility).

    Returns:
        1.0 if all hard constraints pass, 0.0 otherwise.
    """
    _ensure_rdkit()

    smiles = _extract_smiles(output)
    if not smiles:
        return 0.0

    return 1.0 if _check_hard_constraints(smiles, reference_smiles) else 0.0
