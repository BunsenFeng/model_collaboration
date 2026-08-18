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

    smiles = output.strip().split()[0].strip('`"\'') if output.strip() else ""
    if not smiles:
        return 0.0

    return 1.0 if _check_hard_constraints(smiles, reference_smiles) else 0.0
