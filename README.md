# model_collaboration

The future is now.

### Contributors who just joined us, read this

Thank you for joining this effort! Kindly do the following:
1. Get familiar with "Github Setup" in this readme: you need to create your own branch, code in your branch, and submit pull requests from your branch to the dev branch. **Never edit anything on the dev branch.**
2. Follow "After that, quick start!!" in this readme.
3. Read the beginning of `model_collaboration/method/user_readme.md` and briefly check out the methods we already have here.
4. Follow `model_collaboration/method/sample_approach.py` and `model_collaboration/method/developer_readme.md` to develop your own method. Try to use the helper functions provided there if possible (such as `distributed_generation.distributed_generation`).
5. After you implemented and tested your method, git add commit push, open a pull request from your branch to dev, assign your point of contact and Shangbin as the two reviewers.

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

Questions about git? Don't take guesses, email `svenyan234@gmail.com` and cc `bunsenfeng@gmail.com`.

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
python -m model_collaboration.main -c model_collaboration/test_config.json
```

You will see the outputs and evaluation results in the `model_collaboration/logs/` folder.

See `model_collaboration/method/user_readme.md` for more details about different collaboration methods implemented.

Zhaoxuan (our evaluation tsar), additionally see `model_collaboration/data/eval_readme.md`.
