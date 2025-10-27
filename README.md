# model_collaboration

The future is now.

### Quick Start

1. Clone the repo with `https://github.com/BunsenFeng/model_collaboration.git`.
2. Checkout to the dev branch: `git checkout dev`
3. Pull the latest changes from the dev branch to your local dev branch: `git pull`
4. Create your own feature/hotfix branch on local: `git checkout -b [your-local-branch-name]`
5. Make edtis on the scripts you care.
6. Push any changes you made on your local branch to the GitHub server - after `git add` and `git commit` operations, do `git push`, you will see `git push --set-upstream origin [your-local-branch-name]` suggested by github, copy and paste this command and run.
7. Open a new Pull Request from the GitHub webpage, **make sure it's merging from `[your-local-branch-name]` to the `dev` branch**. Add any reviewer and Shangbin that matters to the changes.
8. Once approved, merge the changes to the `dev` branch.
9. After merging, you will see an option on the webpage to delete your own branch. Delete it.
10. Loop from #2.

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
