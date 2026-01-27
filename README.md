# MoCo

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
`MoCo` currently supports the following model collaboration algorithms, across [API-level, text-level, logit-level, and weight-level collaboration](https://arxiv.org/abs/2502.04506). We provide a sample config for each method in `examples/` and please check out `docs/user_readme.md` for more details about writing configs and different collaboration methods implemented.

| **Method** | **Core Idea** | **Code** | **Sample Config** | **Doc** |
|------------|---------------|----------|-------------------|---------|
| API: Nudging | one model guides the coding of another | [link](model_collaboration/method/api_nudging.py) | [link](examples/api_nudging.json) | [link](docs/user_readme.md#api-level-nudging) |