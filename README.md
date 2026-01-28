<div align="center">
  <img src="docs/MoCo.png" alt="MoCo Logo" width="200">
</div>

## MoCo: A One-Stop Shop for Model Collaboration Research

`MoCo` is a toolkit for **Mo**del **Co**llaboration research, where multiple language models collaborate and complement each other for compositional AI systems.

## Quick Start

```
conda env create -f environment.yml
pip install modelco
conda activate model_collaboration
```

Run your first model collaboration experiment (if you don't have 3 GPUs, go to `model_collaboration/test_config.json` and set `"gpu_ids": [0]`, `[0,1]`, or whatever you have; if your GPU is nice, increase `batch_size`):

```
python -m model_collaboration.main -c model_collaboration/test_config.json
```

You will see the outputs and evaluation results in the `model_collaboration/logs/` folder.

## Supported Methods
`MoCo` currently supports the following model collaboration algorithms, across [API-level, text-level, logit-level, and weight-level collaboration](https://arxiv.org/abs/2502.04506). We provide a sample config for each method in `examples/` and please check out `docs/user_readme.md` for more details about writing configs and the different collaboration methods implemented.

| **Method** | **Core Idea** | **Code** | **Sample Config** | **Doc** |
|------------|---------------|----------|-------------------|---------|
| API: Nudging | one model guides the decoding of another | [link](model_collaboration/method/api_nudging.py) | [link](examples/api_nudging.json) | [link](docs/user_readme.md#api-level-nudging) |
| API: Prompt Routing | prompt an LM to decide which model to use based on model descriptions | [link](model_collaboration/method/api_prompt_routing.py) | [link](examples/api_prompt_routing.json) | [link](docs/user_readme.md#api-level-prompt-routing) |
| API: Switch Generation | multiple LMs take turns to generate parts of the response | [link](model_collaboration/method/api_switch_generation.py) | [link](examples/api_switch_generation.json) | [link](docs/user_readme.md#api-level-switch-generation) |
| API: Trained Router | train an LM to route based on the dev set | [link](model_collaboration/method/api_trained_router.py) | [link](examples/api_trained_router.json) | [link](docs/user_readme.md#api-level-trained-router) |
| API: Graph Routing | train a graph neural network for routing | [link](model_collaboration/method/api_graph_routing.py) | [link](examples/api_graph_routing.json) | [link](docs/user_readme.md#api-level-graph-routing) |
| API: Cascade | use multiple models in a cascade to improve efficiency | [link](model_collaboration/method/api_cascade.py) | [link](examples/api_cascade.json) | [link](docs/user_readme.md#api-level-cascade) |
| API: Mentor Collab | a mentor model guides a smaller student model for generation | [link](model_collaboration/method/api_mentor_collab.py) | [link](examples/api_mentor_collab.json) | [link](docs/user_readme.md#api-level-mentorcollab) |
| API: Co-LLM | train LMs to defer to another model when uncertain | [link](model_collaboration/method/api_collm.py) | [link](examples/api_collm.json) | [link](docs/user_readme.md#api-level-collm) |
| Text: Multiagent Refine | multiple LMs refine each other's answers iteratively | [link](model_collaboration/method/text_multiagent_refine.py) | [link](examples/text_multiagent_refine.json) | [link](docs/user_readme.md#text-level-multiagent-refine) |
| Text: Multiagent Feedback | multiple LMs provide feedback to each other's answers | [link](model_collaboration/method/text_multiagent_feedback.py) | [link](examples/text_multiagent_feedback.json) | [link](docs/user_readme.md#text-level-multiagent-feedback) |
| Text: Knowledge Card | models generate knowledge paragraphs to assist each other | [link](model_collaboration/method/text_knowledge_card.py) | [link](examples/text_knowledge_card.json) | [link](docs/user_readme.md#text-level-knowledge-card) |
| Text: LLM Blender | use ranker and fuser LMs to combine multiple answers | [link](model_collaboration/method/text_llm_blender.py) | [link](examples/text_llm_blender.json) | [link](docs/user_readme.md#text-level-llm-blender) |
| Text: Heterogeneous Swarms | optimize a graph of multiple LLMs for collaboration | [link](model_collaboration/method/text_heterogeneous_swarms.py) | [link](examples/text_heterogeneous_swarms.json) | [link](docs/user_readme.md#text-level-heterogeneous-swarms) |
| Text: Majority Vote | majority vote | [link](model_collaboration/method/text_majority_vote.py) | [link](examples/text_majority_vote.json) | [link](docs/user_readme.md#text-level-majority-vote) |
| Text: Structured Interaction | execute a structured interaction protocol among LLMs | [link](model_collaboration/method/text_structure.py) | [link](examples/text_structure.json) | [link](docs/user_readme.md#text-level-structured-interaction) |
| Text: Multiagent Finetuning | multiple LLMs critique, debate, and refine via finetuning | [link](model_collaboration/method/text_multiagent_finetuning.py) | [link](examples/text_multiagent_finetuning.json) | [link](docs/user_readme.md#text-level-multiagent-finetuning) |
| Text: BBMAS | blackboard-based collaboration among LLMs | [link](model_collaboration/method/text_bbmas.py) | [link](examples/text_bbmas.json) | [link](docs/user_readme.md#text-level-blackboard-multi-agent-system-bbmas) |
| Text: Sparta Alignment | models compete and combat for collective alignment | [link](model_collaboration/method/text_sparta.py) | [link](examples/text_sparta.json) | [link](docs/user_readme.md#text-level-sparta) |
| Text: AggLM | RL to train a solution aggregation model | [link](model_collaboration/method/text_agglm.py) | [link](examples/text_agglm.json) | [link](docs/user_readme.md#text-level-agglm) |
| Logit: Logit Fusion | merge the next-token logits from multiple models | [link](model_collaboration/method/logit_logit_fusion.py) | [link](examples/logit_logit_fusion.json) | [link](docs/user_readme.md#logit-level-logit-fusion) |
| Logit: Logit Contrastive | contrast the logits from best/worst models | [link](model_collaboration/method/logit_logit_contrastive.py) | [link](examples/logit_logit_contrastive.json) | [link](docs/user_readme.md#logit-level-logit-contrastive) |
| Weight: Greedy Soup | iteratively consider adding each model's weights from best to worst | [link](model_collaboration/method/weight_greedy_soup.py) | [link](examples/weight_greedy_soup.json) | [link](docs/user_readme.md#weight-level-greedy-soup) |
| Weight: Dare Ties | the dare-ties model merging algorithm | [link](model_collaboration/method/weight_dare_ties.py) | [link](examples/weight_dare_ties.json) | [link](docs/user_readme.md#weight-level-dare-ties) |
| Weight: Model Swarms | particle swarm optimization for models to search in the weight space | [link](model_collaboration/method/weight_model_swarms.py) | [link](examples/weight_model_swarms.json) | [link](docs/user_readme.md#weight-level-model-swarms) |
| Weight: LoraHub | gradient-free optimization of lora combinations | [link](model_collaboration/method/weight_lorahub.py) | [link](examples/weight_lorahub.json) | [link](docs/user_readme.md#weight-level-lorahub) |
| Weight: ExPO | model weight extrapolation | [link](model_collaboration/method/weight_expo.py) | [link](examples/weight_expo.json) | [link](docs/user_readme.md#weight-level-expo-model-extrapolation) |

Please note that MoCo does not aim to be a reproducibility study: we adapt the core ideas behind related papers and employ what works flexibly.

## Supported Data
`MoCo` comes with a lot of evaluation datasets built-in, and you are free to bring your own datasets, or even just generate responses only and take evaluation elsewhere. Essentially, change the `task` and `task_type` in the config to use diverse datasets. Check out [link](docs/eval_readme.md) for more details.

## Contributing to MoCo
We welcome contributions to `MoCo`!

If you are interested in contributing new model collaboration methods, check out [link](docs/developer_readme.md).

If you are interested in contributing new datasets, check out [link](docs/eval_readme.md).

If you have any suggestions, please open an issue.

## Citation
If `MoCo` is helpful for you, please consider citing:

```
@article{feng2025one,
  title={When one llm drools, multi-llm collaboration rules},
  author={Feng, Shangbin and Ding, Wenxuan and Liu, Alisa and Wang, Zifeng and Shi, Weijia and Wang, Yike and Shen, Zejiang and Han, Xiaochuang and Lang, Hunter and Lee, Chen-Yu and others},
  journal={arXiv preprint arXiv:2502.04506},
  year={2025}
}

// MoCo reference pending
```

Also, please cite the related papers for the methods you employed, as listed in `docs/user_readme.md`.

Have a nice day.