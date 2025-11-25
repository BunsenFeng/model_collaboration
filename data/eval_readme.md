Hi, Zhaoxuan, this is not a serious doc, it's more like a notepad for you.

I have implemented a bunch of quick datasets and evaluations. You will add more later. But for the first phase, help me check if they are all right.

Go look at `README.md` and `method/user_readme.md` first.

You can find the code in `data/eval.py`. The evaluation logic mainly involves two parts: loading datasets and evaluating results.

`prepare_inputs(task, task_type, split)`: given `task` and `task_type`, given the `split` (`dev` or `test`), return the inputs for generation as a list.

`get_scores(task, task_type, split, outputs)`: given `task`, `task_type`, `split`, and the generated `outputs`, return the evaluation scores as a list.

Then, here are all the combinations of `(task, task_type)` that I have implemented for evaluation so far.

```
(agieval, multiple_choice)
(arc_challenge, multiple_choice)
(mmlu_redux, multiple_choice)
(bbh, exact_match)
(gsm8k, exact_match)
(math, exact_match)
(wikidyk, f1_match)
(sciencemeter, multiple_choice)
(popqa, f1_match)
(blend, multiple_choice)
(truthfulqa, multiple_choice) # yes there is a multiple choice format for this and I find it easy to eval
(coconot, noncompliance) # a special type
(alpacaeval, reward_model)
(wildchat, reward_model)
(sciriff, exact_match)
(culturebench, exact_match)
(human_interest, reward_model) # the three instruction following datasets are all evaluated by a reward model now. should we change to something else? up to you
(tablemwp_multiple_choice, multiple_choice)
(tablemwp_free_text, exact_match)
(medmcqa, multiple_choice)
(medqa, multiple_choice)
(pubmedqa, exact_match)
```

Additionally, there is a `text_generation` task type, for people that just want to generate text without evaluation or do their custom evaluation elsewhere. You can try `(wildchat, text_generation)` for that. Essentially any data JSON file with an `input` field can be used with this task type.

Your job in phase 1 is to try **every single one of them** to see if the evaluation logic is correct. You could use the `text_multiagent_refine` approach to generate the outputs, no need to go through all approaches (others will). Thank you! Consult with me and open a pull request for any changes to `data/eval.py` or `data/` in general if you find any issues. 