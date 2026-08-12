"""
Pharmacophore evaluation for SMDD-Bench Type-1 tasks.

The model submits a Python function check_pharmacophore(smiles) -> bool.
We run it against hidden actives (expect True) and hidden decoys (expect False),
and score as balanced accuracy: 0.5 * recall + 0.5 * specificity.

Model code is executed in an isolated subprocess via execute_code_safely (the same
sandbox used for the coding task type) with a 60-second timeout and OS-level
resource limits — preventing infinite loops, network access, and filesystem writes.
RDKit SMILES validation runs in-process (trusted code only).
"""

import importlib
import json
import subprocess
import sys

# Per-submission timeout passed to execute_code_safely.
PHARMACOPHORE_TIMEOUT = 60


def _ensure_rdkit():
    if importlib.util.find_spec("rdkit") is None:
        print("[pharmacophore_eval] rdkit not found, installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "rdkit", "-q"]
        )


def _valid_smiles(smiles_list: list[str]) -> list[str]:
    """Filter to chemically valid SMILES using in-process RDKit."""
    from rdkit import Chem
    return [s for s in smiles_list if s and Chem.MolFromSmiles(s) is not None]


def score_pharmacophore(code: str, hidden_actives: list[str], hidden_decoys: list[str]) -> float:
    """
    Score a model's check_pharmacophore implementation.

    Model code runs in a sandboxed subprocess with a timeout. Recall and
    specificity are computed from the subprocess output.

    Args:
        code:           Full Python source code emitted by the model.
        hidden_actives: SMILES strings that should return True.
        hidden_decoys:  SMILES strings that should return False.

    Returns:
        float in [0, 1]: 0.5 * recall + 0.5 * specificity.
        0.0 if the code fails, times out, or does not define check_pharmacophore.
    """
    from model_collaboration.data.eval import execute_code_safely

    _ensure_rdkit()

    actives = _valid_smiles(hidden_actives)
    decoys = _valid_smiles(hidden_decoys)

    if not actives and not decoys:
        return 0.0

    # Build a self-contained runner that embeds the SMILES data as literals,
    # calls check_pharmacophore on each, and prints JSON results to stdout.
    actives_json = json.dumps(actives)
    decoys_json = json.dumps(decoys)

    runner = f"""\
import json

{code}

if 'check_pharmacophore' not in dir():
    print(json.dumps({{"error": "check_pharmacophore not defined"}}))
else:
    actives = {actives_json}
    decoys  = {decoys_json}

    recall_hits = sum(1 for s in actives if check_pharmacophore(s) is True)
    spec_hits   = sum(1 for s in decoys  if check_pharmacophore(s) is False)

    print(json.dumps({{
        "recall":      recall_hits / len(actives) if actives else 0.0,
        "specificity": spec_hits   / len(decoys)  if decoys  else 0.0,
    }}))
"""

    success, stdout, stderr = execute_code_safely(runner, "", PHARMACOPHORE_TIMEOUT)

    if not success or not stdout.strip():
        return 0.0

    try:
        result = json.loads(stdout.strip().splitlines()[-1])
        if "error" in result:
            return 0.0
        recall = float(result.get("recall", 0.0))
        specificity = float(result.get("specificity", 0.0))
        return 0.5 * recall + 0.5 * specificity
    except Exception:
        return 0.0
