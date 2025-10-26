# model_collaboration

The future is now.

### Quick Start

Use the dev branch! Do `git checkout dev` please.

```
conda env create -f environment.yml
conda activate model_collaboration
```

Run your first model collaboration experiment (if you don't have 3 GPUs, go to `test_config.json` and set `"gpu_ids": [0]`, `[0,1]`, or whatever you have):

```
python main.py -c test_config.json
```

You will see the outputs and evaluation results in the `logs/` folder.

See `method/user_readme.md` for more details about different collaboration methods implemented.

Zhaoxuan (our evaluation tsar), additionally see `data/eval_readme.md`.
