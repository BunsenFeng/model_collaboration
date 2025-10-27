# model_collaboration

The future is now.

### Github Setup

1. Clone the repo with `https://github.com/BunsenFeng/model_collaboration.git`.
2. Checkout to the dev branch: `git checkout dev`
3. Pull the latest changes from the dev branch to your local dev branch: `git pull`
4. Create your own feature/hotfix branch on local: `git checkout -b [your-local-branch-name]`
5. Make edits on the scripts you care.
6. Push any changes you made on your local branch to the GitHub server - after `git add` and `git commit` operations, do `git push`, you will see `git push --set-upstream origin [your-local-branch-name]` suggested by github, copy and paste this command and run.
7. Open a new Pull Request from the GitHub webpage, **make sure it's merging from `[your-local-branch-name]` to the `dev` branch**. Add any reviewer and Shangbin that matters to the changes.
8. Once approved, merge the changes to the `dev` branch.
9. After merging, you will see an option on the webpage to delete your own branch. Delete it.
10. Loop from #2.

If you are in the middle of the development, and you need the latest changes from dev branch, follow the steps below:
1. Keep track of the current changes you made on your local branch: `git add` and `git commit` your `[your-local-branch-name]`
2. Checkout to the dev branch: `git checkout dev`
3. Pull the latest changes from the dev branch: `git pull`
4. Check back to your local branch: `git checkout [your-local-branch-name]`
5. Merge the changes from dev branch to your own branch: `git merge dev`
6. Keep working on your own branch. done.

### After that, quick start!!

```
conda env create -f environment.yml
conda activate model_collaboration
cd ..
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit
pip install -e .
cd ..
cd model_collaboration
```

Run your first model collaboration experiment (if you don't have 3 GPUs, go to `test_config.json` and set `"gpu_ids": [0]`, `[0,1]`, or whatever you have; if your GPU is nice, increase `batch_size`):

```
python main.py -c test_config.json
```

You will see the outputs and evaluation results in the `logs/` folder.

See `method/user_readme.md` for more details about different collaboration methods implemented.

Zhaoxuan (our evaluation tsar), additionally see `data/eval_readme.md`.
