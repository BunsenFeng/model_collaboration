"""
Pharmacophore evaluation for SMDD-Bench Type-1 tasks.

The model submits a Python function check_pharmacophore(smiles) -> bool.
We run it against hidden actives (expect True) and hidden decoys (expect False),
and score as balanced accuracy: 0.5 * recall + 0.5 * specificity.

RDKit is used only to validate SMILES before passing them to the model's function.
It is not in the base MoCo image, so we install it on first use.
"""

import importlib
import subprocess
import sys


def _ensure_rdkit():
    if importlib.util.find_spec("rdkit") is None:
        print("[pharmacophore_eval] rdkit not found, installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "rdkit", "-q"]
        )


def _load_check_fn(code: str):
    """Exec model code and return its check_pharmacophore function, or None."""
    ns = {}
    try:
        exec(compile(code, "<model_solution>", "exec"), ns)
    except Exception as e:
        print(f"[pharmacophore_eval] compile/exec error: {e}")
        return None
    fn = ns.get("check_pharmacophore")
    if fn is None:
        print("[pharmacophore_eval] check_pharmacophore not found in model output")
    return fn


def _run_against_smiles(fn, smiles_list: list[str], expect_true: bool) -> float:
    """
    Run fn over smiles_list. Returns fraction of valid molecules where fn
    returns expect_true.  Exceptions count as wrong (not as correct rejections).
    """
    from rdkit import Chem

    hits = 0
    valid = 0
    for smi in smiles_list:
        if not smi:
            continue
        if Chem.MolFromSmiles(smi) is None:
            continue
        valid += 1
        try:
            result = fn(smi)
            if bool(result) is expect_true:
                hits += 1
        except Exception:
            pass
    return hits / valid if valid > 0 else 0.0


def score_pharmacophore(code: str, hidden_actives: list[str], hidden_decoys: list[str]) -> float:
    """
    Score a model's check_pharmacophore implementation.

    Args:
        code:           Full Python source code emitted by the model.
        hidden_actives: SMILES strings that should return True.
        hidden_decoys:  SMILES strings that should return False.

    Returns:
        float in [0, 1]: 0.5 * recall + 0.5 * specificity.
        0.0 if the code fails to load or define check_pharmacophore.
    """
    _ensure_rdkit()

    fn = _load_check_fn(code)
    if fn is None:
        return 0.0

    recall = _run_against_smiles(fn, hidden_actives, expect_true=True)
    specificity = _run_against_smiles(fn, hidden_decoys, expect_true=False)
    return 0.5 * recall + 0.5 * specificity
