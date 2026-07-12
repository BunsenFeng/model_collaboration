"""
Scoring utilities for AssayBench gene ranking evaluation.

Extracted from https://github.com/genentech/AssayBench (MIT License).
Gene normalization is intentionally skipped; predicted genes are uppercased only.

Primary metric: Adjusted nDCG@100 (AnDCG@100), a scalar in [0, 1].
"""

import re
import numpy as np
from typing import List, Optional
from scipy.stats import rankdata


def hit_scale_relevance_scores(relevance_scores: List[float]) -> List[float]:
    """Rescale relevance scores to within-hit percentile ranks."""
    scores = np.array(relevance_scores, dtype=float)

    pos_mask = scores > 0
    n_pos = int(pos_mask.sum())
    if n_pos > 0:
        ranks = rankdata(-scores[pos_mask], method="average")
        scores[pos_mask] = 1 - (ranks - 1) / n_pos

    neg_mask = scores < 0
    n_neg = int(neg_mask.sum())
    if n_neg > 0:
        ranks = rankdata(scores[neg_mask], method="average")
        scores[neg_mask] = -(1 - (ranks - 1) / n_neg)

    return scores.tolist()


def extract_genes_from_output(output: str) -> List[str]:
    """Parse comma-separated gene list from model output, uppercased."""
    match = re.search(r'<Final Answer>(.*?)</Final Answer>', output, re.DOTALL | re.IGNORECASE)
    genes_text = match.group(1) if match else output
    genes = [g.strip().upper() for g in genes_text.split(',')]
    return [g for g in genes if g]


def _compute_dcg(relevances: List[Optional[float]], k: int) -> float:
    padded = list(relevances) + [0.0] * max(0, k - len(relevances))
    condensed = [r for r in padded[:k] if r is not None][:k]
    if not condensed:
        return 0.0
    return sum(r / np.log2(i + 2) for i, r in enumerate(condensed))


def _compute_ndcg(predicted_relevances, all_relevances, k: int) -> float:
    dcg = _compute_dcg(predicted_relevances, k)
    ideal = sorted([max(r, 0.0) for r in all_relevances], reverse=True)
    idcg = _compute_dcg(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def _compute_adjusted_ndcg(predicted_relevances, all_relevances, k: int) -> float:
    ndcg = _compute_ndcg(predicted_relevances, all_relevances, k)
    ndcg_rand = _compute_ndcg(
        [np.mean(all_relevances)] * min(k, len(all_relevances)),
        all_relevances, k
    )
    if 1 - ndcg_rand == 0:
        return 0.0
    return max((ndcg - ndcg_rand) / (1 - ndcg_rand), 0.0)


def score_gene_ranking(
    output: str,
    relevance_genes: List[str],
    relevance_scores: List[float],
    k: int = 100,
) -> float:
    """
    Score a model's gene ranking output against ground truth.

    Args:
        output: Raw model output (comma-separated gene list).
        relevance_genes: Ground truth gene names.
        relevance_scores: Relevance score per gene (0 if non-hit, >0 if hit).
        k: Cutoff for AnDCG (default 100).

    Returns:
        AnDCG@k score in [0, 1].
    """
    predicted_genes = list(dict.fromkeys(extract_genes_from_output(output)))
    gt_genes_upper = [g.strip().upper() for g in relevance_genes]
    scoring_dict = dict(zip(gt_genes_upper, relevance_scores))

    predicted_relevances = [scoring_dict.get(g) for g in predicted_genes]

    return _compute_adjusted_ndcg(predicted_relevances, relevance_scores, k)
