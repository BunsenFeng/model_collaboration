"""
Multiagent Finetuning method implementation.

This method implements a simplified version of the multiagent finetuning
procedure described in the paper "Multiagent Finetuning: Self Improvement
with Diverse Reasoning Chains" (ICLR 2025).  The high level idea is to run
multiple language models in a debate setting, construct task‑specific
supervised fine–tuning (SFT) datasets from their responses, fine–tune each
model independently on its own dataset, and then iterate this procedure
several times.  At inference time the finetuned models again engage in a
debate and the final answer is selected by majority vote.  By training on
disjoint datasets derived from the debate, each model can specialise on
different reasoning chains which leads to improved diversity and accuracy.

Method‑specific hyperparameters supported:

    iterations: int (default 1)
        Number of finetuning iterations.  Each iteration performs a debate on
        the dev set, constructs new SFT datasets and fine–tunes the models.

    rounds: int (default 2)
        Number of debate rounds.  Round 0 is generation (initial answer),
        subsequent rounds are critic rounds.  The default of 2 corresponds
        to one generation round followed by one critic round.

    w: float in [0,1] (default 0.5)
        Weighting for critic SFT data.  A fraction `w` of the critic
        dataset is sampled from examples where the initial answer was
        incorrect but the final answer matched the debate's consensus
        (DC⁻), and a fraction (1‑w) is sampled from examples where the
        initial answer already matched the consensus and remained correct
        throughout the debate (DC⁺).

    training_ratio: float in (0,1] (default 1.0)
        Fraction of the dev set to use for constructing SFT datasets.  This
        can be set to a smaller value to reduce compute.

    sft_epochs: int (default 1)
        Number of epochs for LoRA fine–tuning.  See `utils/distributed_sft.py`.

    sft_learning_rate: float (default 1e-5)
        Learning rate for LoRA fine–tuning.

    sft_batch_size: int (default 1)
        Per device batch size for LoRA fine–tuning.

    sft_grad_accum: int (default 16)
        Gradient accumulation steps for LoRA fine–tuning.
"""

import os
import json
import random
from collections import Counter
from typing import List, Dict, Tuple, Any

from model_collaboration.data import eval

# Directory where the evaluation data JSON files reside.  This mirrors the
# convention used in text_majority_vote.py.  We rely on this constant
# when loading the raw data for answer extraction during the debate and
# evaluation phases.  Without access to the original questions and
# multiple‑choice options it is not possible to accurately extract the
# model's chosen answer.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
from model_collaboration.method import distributed_generation
from model_collaboration.utils import distributed_sft


def _majority_vote(responses: List[str]) -> str:
    """Return the string that appears most frequently in the list.
    Break ties deterministically by lexicographic order.
    """
    counts = Counter()
    for r in responses:
        if r is None:
            continue
        counts[r.strip()] += 1
    if not counts:
        return ""
    # sort by frequency descending then lexicographically
    sorted_items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return sorted_items[0][0]

# -----------------------------------------------------------------------------
# Summarisation utilities
#
# In the original multiagent finetuning algorithm, critic agents receive a
# *summary* of the responses from the other agents before generating their own
# critique【889394202153893†L376-L384】.  The summarisation step removes redundant
# information and highlights the key differences in the other agents’ answers【889394202153893†L584-L589】.
# We implement a simple summarisation function here.  For each set of
# responses we concatenate the responses with double newlines.  Users can
# override this function to call a more sophisticated summarisation model.

def _summarize_other_answers(other_answers: List[str]) -> str:
    """Produce a simple summary of other agents' answers.

    Given a list of answer strings from other agents, join them with
    double newlines to create a consolidated summary.  This naive approach
    preserves the content of each response while reducing redundancy.  If a
    more sophisticated summarisation is desired, this function can be
    replaced by a call to a summarisation model.

    Args:
        other_answers: List of responses from other agents.

    Returns:
        A single string summarising the other responses.
    """
    # Strip and deduplicate empty answers
    cleaned = [ans.strip() for ans in other_answers if ans and ans.strip()]
    if not cleaned:
        return ""
    return "\n\n".join(cleaned)


def _build_critic_prompt_with_summary(question: str, summary: str, self_answer: str) -> str:
    """Construct a prompt for the critic model using a summary of other agents' answers.

    The prompt includes the question, a summary of the other agents' answers
    (provided by `_summarize_other_answers`), and the critic's own answer
    from the previous round.  The critic should use this information to
    produce a refined and correct answer.

    Args:
        question: The original question string.
        summary: A summary of other agents' answers.
        self_answer: The critic's own answer from the previous round.

    Returns:
        A prompt string for critic generation.
    """
    lines = []
    lines.append(f"Question: {question}")
    lines.append("Summary of other assistants' answers:")
    lines.append(summary)
    lines.append(f"Your previous answer: {self_answer}")
    lines.append("Please provide a refined and correct answer to the question.")
    return "\n".join(lines)


def _build_critic_prompt(question: str, other_answers: List[str], self_answer: str) -> str:
    """
    Construct a simple prompt for the critic model.  The prompt includes the
    question, a list of other agents' answers, and the model's own initial
    answer.  The critic should produce a refined and correct answer based on
    these inputs.
    """
    prompt_lines = []
    prompt_lines.append(f"Question: {question}")
    prompt_lines.append("The following are answers provided by other assistants:")
    for ans in other_answers:
        prompt_lines.append(f"- {ans}")
    prompt_lines.append(f"Your initial answer: {self_answer}")
    prompt_lines.append("Please provide a refined and correct answer to the question.")
    return "\n".join(prompt_lines)


def _save_jsonl(data: List[Dict[str, str]], path: str) -> None:
    """Write a list of {"prompt": ..., "completion": ...} dicts to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for entry in data:
            # ensure keys exist
            prompt = entry.get("prompt", "")
            completion = entry.get("completion", "")
            f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")


def run_method(task: str,
               task_type: str,
               gpu_ids: List[int],
               model_names: List[str],
               hyperparameters: Dict[str, Any]) -> int:
    """
    Run the multiagent finetuning method on a given task.

    Args:
        task: name of the evaluation dataset (e.g. 'math', 'agieval').
        task_type: type of the task (e.g. 'exact_match', 'multiple_choice').
        gpu_ids: list of GPU indices available for generation and SFT.
        model_names: list of base model identifiers to use as agents.  These
            should be loadable via `transformers.AutoModelForCausalLM.from_pretrained()`.
        hyperparameters: dictionary of method specific and general
            hyperparameters.  See module docstring for supported keys.

    Returns:
        0 upon successful completion.
    """

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    # Extract method‑specific hyperparameters with sensible defaults
    iterations = int(hyperparameters.get("iterations", 1))
    rounds = int(hyperparameters.get("rounds", 2))
    w = float(hyperparameters.get("w", 0.5))
    training_ratio = float(hyperparameters.get("training_ratio", 1.0))
    # SFT hyperparameters
    sft_epochs = int(hyperparameters.get("sft_epochs", 1))
    sft_learning_rate = float(hyperparameters.get("sft_learning_rate", 1e-5))
    sft_batch_size = int(hyperparameters.get("sft_batch_size", 1))
    sft_grad_accum = int(hyperparameters.get("sft_grad_accum", 16))

    # ------------------------------------------------------------------
    # Prepare the development inputs and corresponding raw data items.
    #
    # We need access to the underlying data (e.g. the multiple‑choice
    # options or the ground truth answers) in order to extract the
    # model's answer from its raw response.  The eval.prepare_inputs
    # function returns only the formatted question strings.  Therefore
    # we load the original JSON file for the selected task and split,
    # and take a matching subset when training_ratio < 1.0.
    # ------------------------------------------------------------------
    # Load the full dev split data
    with open(os.path.join(DATA_DIR, f"{task}.json"), "r") as f_data:
        full_data = json.load(f_data)
    dev_data_full = full_data.get("dev", [])
    # Format questions using helper
    dev_inputs_full = eval.prepare_inputs(task, task_type, "dev")
    assert len(dev_inputs_full) == len(dev_data_full), "Mismatch between dev inputs and data length"
    if training_ratio < 1.0:
        n_total = len(dev_inputs_full)
        n_keep = max(1, int(n_total * training_ratio))
        random.seed(42)
        selected_indices = sorted(random.sample(range(n_total), n_keep))
        dev_inputs = [dev_inputs_full[i] for i in selected_indices]
        dev_data = [dev_data_full[i] for i in selected_indices]
    else:
        dev_inputs = dev_inputs_full
        dev_data = dev_data_full

    # Track the current generation and critic model paths.  Initially they
    # correspond to the base models provided by the user.  After each
    # iteration they will be replaced by the newly fine–tuned models.
    current_generation_models = list(model_names)
    current_critic_models = list(model_names)

    # Directory to store intermediate data and models
    method_name = "multiagent_finetuning"
    base_log_dir = os.path.join("logs", method_name)

    # Iterate the finetuning loop
    for it in range(iterations):
        print(f"\nStarting finetuning iteration {it+1}/{iterations}")

        # --------------------------------------------------------------
        # Multiagent debate on the dev set.
        #
        # We maintain two parallel structures: `answers`, which stores
        # the raw string responses from each agent at each round, and
        # `extracted_answers_all`, which stores the canonical extracted
        # answers (e.g. the letter for multiple‑choice or the boxed
        # expression for math).  The extracted answers are used for
        # majority voting and for determining which examples to include
        # in the SFT datasets.  This separation allows us to retain the
        # original responses for use as completions while basing the
        # selection logic on the content of the answer rather than
        # formatting or extraneous explanation.
        # --------------------------------------------------------------
        N = len(current_generation_models)
        # Generate initial answers (round 0)
        # Build list_of_input_list: replicate dev_inputs for each model
        list_of_input_list = [dev_inputs for _ in range(N)]
        gen_outputs = distributed_generation.distributed_generation(
            current_generation_models,
            list_of_input_list,
            gpu_ids
        )  # shape: N x len(dev_inputs)
        # Extract answers for round 0
        extracted_round = []  # will be list of length N, each a list of len(dev_inputs)
        for n in range(len(current_generation_models)):
            model_outputs = gen_outputs[n]
            model_extracted = []
            # iterate through outputs and corresponding data items
            for out, item in zip(model_outputs, dev_data):
                if task_type == "multiple_choice":
                    # Build options list preserving order defined in input JSON
                    options = []
                    # choices is a dict mapping option letter to option text
                    for key in item["choices"].keys():
                        options.append(item["choices"][key])
                    letter, _ = eval.parse_model_response_mcq(out, options)
                    model_extracted.append(letter)
                elif task_type in ["exact_match", "f1_match"]:
                    model_extracted.append(eval.extract_answer_text(out))
                else:
                    # For unsupported task types, fall back to the raw response
                    model_extracted.append(out.strip())
            extracted_round.append(model_extracted)

        # Initialize list of extracted answers per round.  Start with
        # round 0 extractions.
        extracted_answers_all = []
        extracted_answers_all.append(extracted_round)

        # If more than one round, run critic rounds.  We support generic
        # number of rounds.  For m>=1 we treat all models as critics and
        # generate refined answers conditioned on other agents' responses.
        # We maintain a list of answers per model per round.
        # answers[m][n][i] = answer from agent n at round m for question i
        answers = []
        answers.append(gen_outputs)

        # For each subsequent round
        for m in range(1, rounds):
            print(f"Generating critic answers for round {m+1}/{rounds}")
            list_of_input_list_round = []
            # Construct critic prompts for each model
            for n in range(N):
                prompts = []
                for idx, question in enumerate(dev_inputs):
                    # gather previous round answers from all models
                    prev_answers = answers[m-1]
                    # list of answers at previous round for this question
                    all_prev = [prev_answers[k][idx] for k in range(N)]
                    # other agents' answers (exclude self)
                    other_answers = [all_prev[k] for k in range(N) if k != n]
                    self_ans = all_prev[n]
                    # summarise other answers
                    summary = _summarize_other_answers(other_answers)
                    prompt = _build_critic_prompt_with_summary(question, summary, self_ans)
                    prompts.append(prompt)
                list_of_input_list_round.append(prompts)
            # Generate critic answers with current critic models
            critic_outputs = distributed_generation.distributed_generation(
                current_critic_models,
                list_of_input_list_round,
                gpu_ids
            )  # N x len(dev_inputs)
            answers.append(critic_outputs)
            # Extract answers for this critic round
            extracted_round = []
            for n in range(len(current_critic_models)):
                model_outputs = critic_outputs[n]
                model_extracted = []
                for out, item in zip(model_outputs, dev_data):
                    if task_type == "multiple_choice":
                        options = []
                        for key in item["choices"].keys():
                            options.append(item["choices"][key])
                        letter, _ = eval.parse_model_response_mcq(out, options)
                        model_extracted.append(letter)
                    elif task_type in ["exact_match", "f1_match"]:
                        model_extracted.append(eval.extract_answer_text(out))
                    else:
                        model_extracted.append(out.strip())
                extracted_round.append(model_extracted)
            extracted_answers_all.append(extracted_round)

        # The final round answers and extracted answers
        final_round_answers = answers[-1]  # shape N x len(dev_inputs)
        extracted_final_answers = extracted_answers_all[-1]  # shape N x len(dev_inputs)

        # Majority vote across models based on extracted answers for each question
        # to obtain the consensus.  We also select a representative raw
        # consensus answer among those agents whose extracted answer matches the
        # consensus.  This raw consensus will be used as the completion in
        # critic datasets.  Without this mapping the finetuned critic may not
        # learn the desired formatting (e.g. including \boxed{{...}}).
        consensus_extracted = []  # list of canonical answers
        consensus_raw = []        # list of representative raw answers
        for idx in range(len(dev_inputs)):
            # gather extracted answers for this question
            extracted_list = [extracted_final_answers[n][idx] for n in range(N)]
            # majority vote ignoring None
            consensus_e = _majority_vote(extracted_list)
            consensus_extracted.append(consensus_e)
            # collect raw answers from agents whose extracted answer matches consensus_e
            raw_candidates: Dict[str, int] = {}
            for n in range(N):
                if extracted_final_answers[n][idx] == consensus_e:
                    raw = final_round_answers[n][idx].strip()
                    raw_candidates[raw] = raw_candidates.get(raw, 0) + 1
            # choose the most frequent raw string, breaking ties lexicographically
            if raw_candidates:
                sorted_raw = sorted(raw_candidates.items(), key=lambda x: (-x[1], x[0]))
                consensus_raw.append(sorted_raw[0][0])
            else:
                # fallback to empty string if no matching raw answer (rare)
                consensus_raw.append("")

        # Construct SFT datasets for generation and critic models
        DG_datasets: List[List[Dict[str, str]]] = [[] for _ in range(N)]
        DC_minus: List[List[Dict[str, str]]] = [[] for _ in range(N)]
        DC_plus: List[List[Dict[str, str]]] = [[] for _ in range(N)]

        # Populate datasets
        for idx, question in enumerate(dev_inputs):
            # canonical answer (for selection) and representative raw answer (for completion)
            c_extracted = consensus_extracted[idx]
            c_raw = consensus_raw[idx]
            for n in range(N):
                # Raw initial answer and its extracted form
                y1_raw = answers[0][n][idx].strip()
                y1_extracted = extracted_answers_all[0][n][idx]
                # Raw final answer and its extracted form
                yM_raw = final_round_answers[n][idx].strip()
                yM_extracted = extracted_final_answers[n][idx]
                # Generation dataset: include only if the extracted initial answer matches the consensus
                if y1_extracted == c_extracted:
                    DG_datasets[n].append({"prompt": question, "completion": y1_raw})
                # Critic dataset: include only if the extracted final answer matches the consensus
                if yM_extracted == c_extracted:
                    # Build a dialogue-style prompt that includes the entire history of the agent's answers
                    history_lines = []
                    for r_i, ans_str in enumerate([answers[k][n][idx] for k in range(len(answers))]):
                        history_lines.append(f"Round {r_i}: {ans_str.strip()}")
                    history_str = "\n".join(history_lines)
                    convo_prompt = f"Question: {question}\n{history_str}\nPlease provide a refined and correct answer to the question."
                    if y1_extracted != c_extracted:
                        # The model changed its answer from incorrect to correct
                        DC_minus[n].append({"prompt": convo_prompt, "completion": c_raw})
                    else:
                        # The model’s initial answer was already correct
                        DC_plus[n].append({"prompt": convo_prompt, "completion": c_raw})

        # Combine DC_minus and DC_plus according to weight w
        DC_datasets: List[List[Dict[str, str]]] = []
        for n in range(N):
            minus_list = DC_minus[n]
            plus_list = DC_plus[n]
            # sample according to w
            random.seed(0)
            selected_entries = []
            if minus_list:
                k_minus = max(1, int(len(minus_list) * w)) if w > 0 else 0
                selected_entries.extend(random.sample(minus_list, min(k_minus, len(minus_list))))
            if plus_list:
                k_plus = max(1, int(len(plus_list) * (1 - w))) if (1 - w) > 0 else 0
                selected_entries.extend(random.sample(plus_list, min(k_plus, len(plus_list))))
            DC_datasets.append(selected_entries)

        # Save datasets to JSONL files
        iter_log_dir = os.path.join(base_log_dir, f"iteration_{it+1}")
        os.makedirs(iter_log_dir, exist_ok=True)
        gen_data_paths = []
        crit_data_paths = []
        for n in range(N):
            gen_path = os.path.join(iter_log_dir, f"gen_data_agent_{n}.jsonl")
            _save_jsonl(DG_datasets[n], gen_path)
            gen_data_paths.append(gen_path)
            crit_path = os.path.join(iter_log_dir, f"critic_data_agent_{n}.jsonl")
            _save_jsonl(DC_datasets[n], crit_path)
            crit_data_paths.append(crit_path)

        # Fine‑tune generation models.  Skip models with empty datasets.
        print("Starting SFT for generation agents...")
        # Paths where the finetuned models would be saved
        gen_output_model_paths = []
        for n in range(N):
            out_dir = os.path.join(iter_log_dir, f"gen_model_agent_{n}")
            gen_output_model_paths.append(out_dir)
        # Prepare lists for models that actually have non‑empty data
        gen_sft_models = []
        gen_sft_data_paths = []
        gen_sft_output_paths = []
        gen_sft_indices = []  # track original indices
        for n in range(N):
            # Only include in SFT if there is at least one example
            if len(DG_datasets[n]) > 0:
                gen_sft_models.append(current_generation_models[n])
                gen_sft_data_paths.append(gen_data_paths[n])
                gen_sft_output_paths.append(gen_output_model_paths[n])
                gen_sft_indices.append(n)
            else:
                # If dataset is empty, skip finetuning.  The model remains unchanged.
                # We still want to have an output path for consistency, but we leave
                # the model path unchanged.  Users can examine logs for details.
                print(f"Skipping SFT for generation agent {n} due to empty dataset.")
        # Run SFT only if there are tasks to perform
        if gen_sft_models:
            distributed_sft.distributed_sft(
                gen_sft_models,
                gen_sft_data_paths,
                gpu_ids,
                gen_sft_output_paths,
                batch_size=sft_batch_size,
                gradient_accumulation_steps=sft_grad_accum,
                learning_rate=sft_learning_rate,
                epoch=sft_epochs
            )
        # Update current generation model paths: assign finetuned paths to those that were trained
        for idx, n in enumerate(gen_sft_indices):
            current_generation_models[n] = gen_sft_output_paths[idx]
        # Fine‑tune critic models.  Skip models with empty datasets.
        print("Starting SFT for critic agents...")
        crit_output_model_paths = []
        for n in range(N):
            out_dir = os.path.join(iter_log_dir, f"critic_model_agent_{n}")
            crit_output_model_paths.append(out_dir)
        crit_sft_models = []
        crit_sft_data_paths = []
        crit_sft_output_paths = []
        crit_sft_indices = []
        for n in range(N):
            if len(DC_datasets[n]) > 0:
                crit_sft_models.append(current_critic_models[n])
                crit_sft_data_paths.append(crit_data_paths[n])
                crit_sft_output_paths.append(crit_output_model_paths[n])
                crit_sft_indices.append(n)
            else:
                print(f"Skipping SFT for critic agent {n} due to empty dataset.")
        if crit_sft_models:
            distributed_sft.distributed_sft(
                crit_sft_models,
                crit_sft_data_paths,
                gpu_ids,
                crit_sft_output_paths,
                batch_size=sft_batch_size,
                gradient_accumulation_steps=sft_grad_accum,
                learning_rate=sft_learning_rate,
                epoch=sft_epochs
            )
        # Update current critic model paths: assign finetuned paths to those that were trained
        for idx, n in enumerate(crit_sft_indices):
            current_critic_models[n] = crit_sft_output_paths[idx]

    # After completing all iterations, evaluate on the test set using the
    # finetuned models in a debate format
    print("\nEvaluating finetuned models on test set...")
    test_inputs = eval.prepare_inputs(task, task_type, "test")
    N = len(current_generation_models)
    # Round 0: generation
    list_of_input_list = [test_inputs for _ in range(N)]
    test_gen_outputs = distributed_generation.distributed_generation(
        current_generation_models,
        list_of_input_list,
        gpu_ids
    )
    # Round m >= 1: critics refine answers
    answers_test = []
    answers_test.append(test_gen_outputs)
    for m in range(1, rounds):
        list_of_input_list_round = []
        for n in range(N):
            prompts = []
            for idx, question in enumerate(test_inputs):
                prev_answers = answers_test[m-1]
                all_prev = [prev_answers[k][idx] for k in range(N)]
                other_answers = [all_prev[k] for k in range(N) if k != n]
                self_ans = all_prev[n]
                summary = _summarize_other_answers(other_answers)
                prompt = _build_critic_prompt_with_summary(question, summary, self_ans)
                prompts.append(prompt)
            list_of_input_list_round.append(prompts)
        critic_outputs_test = distributed_generation.distributed_generation(
            current_critic_models,
            list_of_input_list_round,
            gpu_ids
        )
        answers_test.append(critic_outputs_test)
    final_round_test = answers_test[-1]
    # majority vote per question using extracted answers
    # Load the test data for answer extraction
    test_inputs_list = test_inputs  # alias
    with open(os.path.join(DATA_DIR, f"{task}.json"), "r") as f_data:
        full_data = json.load(f_data)
    test_data = full_data.get("test", [])
    assert len(test_inputs_list) == len(test_data), "Mismatch between test inputs and data length"
    # Extract final answers for each model on the test set
    extracted_final_test = []  # list of N lists
    for n in range(N):
        model_outputs = final_round_test[n]
        model_extracted = []
        for out, item in zip(model_outputs, test_data):
            if task_type == "multiple_choice":
                options = []
                for key in item["choices"].keys():
                    options.append(item["choices"][key])
                letter, _ = eval.parse_model_response_mcq(out, options)
                model_extracted.append(letter)
            elif task_type in ["exact_match", "f1_match"]:
                model_extracted.append(eval.extract_answer_text(out))
            else:
                model_extracted.append(out.strip())
        extracted_final_test.append(model_extracted)
    # Perform majority vote on extracted answers
    final_outputs_extracted: List[str] = []
    for idx in range(len(test_inputs_list)):
        extracted_list = [extracted_final_test[n][idx] for n in range(N)]
        consensus = _majority_vote(extracted_list)
        final_outputs_extracted.append(consensus)
    # Evaluate using extracted answers; eval.get_scores handles parsing internally
    test_scores = eval.get_scores(task, task_type, "test", final_outputs_extracted)
    avg_test_score = sum(test_scores) / len(test_scores) if test_scores else 0.0
    print(f"Final Test {task} score after {iterations} finetuning iteration(s): {avg_test_score}")
    # Log results
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "method": method_name,
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_score,
        "logs": []
    }
    for i in range(len(test_inputs_list)):
        log_entry = {
            "input": test_inputs_list[i],
            "output": final_outputs_extracted[i],
            "score": test_scores[i]
        }
        experiment_logs["logs"].append(log_entry)
    # Save final log file
    final_log_filename = os.path.join(
        "model_collaboration/logs",
        f"{task}_{len(model_names)}_{round(avg_test_score, 4)}_{method_name}.json"
    )
    os.makedirs(os.path.dirname(final_log_filename), exist_ok=True)
    with open(final_log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)
    print(f"Saved experiment logs to {final_log_filename}")
    return 0


if __name__ == "__main__":
    # Example usage (for local testing):
    # Define a minimal config to exercise the code on a tiny dataset.
    example_task = "math"
    example_task_type = "exact_match"
    example_gpu_ids = [0]
    example_model_names = ["gpt2", "gpt2", "gpt2"]
    example_hyperparameters = {
        "iterations": 1,
        "rounds": 2,
        "training_ratio": 0.05,
        "sft_epochs": 1
    }
    run_method(
        example_task,
        example_task_type,
        example_gpu_ids,
        example_model_names,
        example_hyperparameters
    )