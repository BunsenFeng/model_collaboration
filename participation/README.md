# Scaling Participation in Modular AI Systems

This directory contains MoCo configs to replicate the experiments in *Scaling Participation in Modular AI Systems*.

## Setup

Please follow the [main README](../README.md) to install MoCo. Then, from the repository root, run any config with:

```bash
python -m model_collaboration.main -c participation/<path-to-config>.json
```

or, if you installed MoCo with `pip install modelco`:

```bash
moco -c participation/<path-to-config>.json -l ./
```

To run every config in a folder:

```bash
for f in participation/scaling/model_swarms/32/*.json; do python -m model_collaboration.main -c "$f"; done
```

Results are written to `model_collaboration/logs/`. Each config sets `gpu_ids` to `[0,1,2,3]`; edit it to match your hardware.

## Models

- **API- and text-level methods** use the contributed models directly (IDs 1–32 in the model tables of the paper's appendix). Model #4 uses a custom architecture that vLLM does not load, so it is replaced by its distilled copy `bunsenfeng/parti_3_full`.
- **Weight-level methods** use the contributed models distilled into Qwen-2.5-7B: `bunsenfeng/parti_{i}_lora` is contributed model `i+1`.
- A pool of size `k` is always the first `k` models in table order.

## Main results: `scaling/`

These configs reproduce Figure 2 and the main results tables.

| Folder | Experiment |
|---|---|
| `scaling/<method>/<k>/<task>.json` | 14 methods × 15 tasks × k ∈ {2, 4, 8, 16, 32}. Graph routing and heterogeneous swarms start at k=4; multiagent finetuning and AggLM are not run on the instruction-following tasks (AlpacaEval, WildChat, Human Interest). |
| `scaling/baselines/<model>/<task>.json` | Non-participatory baselines: Llama 3.1 (8B, 70B, 405B), Qwen 2.5 (7B, 14B, 32B, 72B), Mixtral (8x7B, 8x22B), DeepSeek-MoE (16B, V2-236B), Marin 8B. |
| `scaling/single_models/model<ID>/<task>.json` | Each of the 32 contributed models on its own. These give the best-single-model baseline and the individual strength in Figure 6. |

## Analyses: `analysis/`

| Folder | Paper |
|---|---|
| `analysis/human_diversity/` | Figure 3: CulturalBench, Kaleidoscope, InfinityChat (open and diversity) with 4 methods, k = 2–32, plus baselines |
| `analysis/model_diversity/` | Figure 4, setting 1: 8-slot pools of the top 1×8, 2×4, 4×2, 8×1 models on AGIEval, BBH, Blend with 4 methods |
| `analysis/compositional_strength/` | Figure 6: 6 methods on AGIEval and BBH with the top-16 pool and each of its 16 leave-one-out pools |
| `analysis/additional_coordination/` | Appendix table on SLM-Mux and Cascade, k = 2–32, on ARC, MATH, TruthfulQA |
| `analysis/ifeval_mixeval/` | Appendix: three strongest methods on IFEval and MixEval, k = 2–32, plus Llama-3.1-405B |
| `analysis/more_diversity/` | Appendix: three strongest methods on MMLU-ProX and CommunityBench, k = 8–32, plus Llama-3.1-405B |
| `analysis/diverse_contributors/` | Appendix: LLM Blender, Sparta alignment, and multiagent finetuning with four models from underrepresented contributor types, on six tasks |

Figure 5 (collaborative emergence) needs no extra configs. It is computed from the per-instance logs of the `scaling/` runs, together with the matching `scaling/single_models/` runs on AGIEval, MMLU-redux, Blend, and TruthfulQA.
