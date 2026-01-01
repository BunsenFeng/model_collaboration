import os
import json
import random
import torch
import shutil
from collections import Counter
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM
import torch.distributed as dist

from model_collaboration.data import eval
from datasets import Dataset
from model_collaboration.method import distributed_generation
from torch import _dynamo


def form_aggregator_instruction(_input, generation_list):
    solution = "\n".join([f"### Solution {idx} begin\n" + gene + f"### Solution {idx} end\n" for idx, gene in enumerate(generation_list)])
    instruction = """Given the following problem:
### Problem begin
{}
### Problem end
and these solution attempts:
{}
It is possible that any, all, or none of these solutions are correct or complete. Carefully review the provided solutions, using them as starting points—correcting mistakes, filling in gaps, and/or combining useful ideas—to produce a final, comprehensive, and correct solution to the problem. Make sure to provide the final answer in the same format as the solutions.
"""
    return instruction.format(_input, solution)


def run_method(task, task_type, gpu_ids, model_names, hyperparameters):

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)

    print("The model you are using are:")
    for model_name in model_names:
        print(model_name)

    # method-specific hyperparameters
    agg_model = hyperparameters.get("agg_model", None)
    if agg_model is None:
        print(
            "Taking Qwen/Qwen3-1.7B as the base model. Please specify the agg_model in hyperparameters to avoid this warning."
        )
        agg_model = 'Qwen/Qwen3-1.7B'

    agglm_log_path = hyperparameters.get("agglm_log_path", 'model_collaboration/logs/agglm')
    reuse_log = hyperparameters.get("reuse_log", True)
    if os.path.exists(agglm_log_path) and not reuse_log:
        shutil.rmtree(agglm_log_path)
    os.makedirs(agglm_log_path, exist_ok=True)
    learning_rate = hyperparameters.get("learning_rate", 1e-4)
    weight_decay = hyperparameters.get("weight_decay", 1e-5)
    lr_scheduler = hyperparameters.get("lr_scheduler", 'cosine')
    max_epoches = hyperparameters.get("max_epoches", 10)
    max_response_length = hyperparameters.get("max_response_length", 512)
    temperature = hyperparameters.get("temperature", 1.0)
    batch_size = hyperparameters.get("train_batch_size", 4)
    s = hyperparameters.get("simple_size", 2)
    m = len(model_names)

    # prepare training data
    file_name = '_'.join([model_name.split("/")[-1] for model_name in model_names]) + f'_{s}.json'
    if not os.path.exists(agglm_log_path + '/' + file_name[:-5] + '/' + 'adapter_model.safetensors'):
        if os.path.exists(agglm_log_path + '/' + file_name):
            dev_res_list = json.load(open(agglm_log_path + '/' + file_name, 'r'))
        else:
            dev_input_list, dev_id_list = eval.prepare_inputs(task, task_type, 'dev', return_id=True)
            dev_res_list = [{'id': _id, 'input': _input, 'generation': []} for _input, _id in zip(dev_input_list, dev_id_list)]
            dev_output_list = distributed_generation.distributed_generation(
                model_names,
                [dev_input_list * s for _ in model_names],
                gpu_ids,
            )
            for idx, data in enumerate(dev_res_list):
                data['generation'] = [dev_output_list[j][i * len(dev_res_list) + idx] for i in range(s) for j in range(m)]
            json.dump(dev_res_list, open(agglm_log_path + '/' + file_name, 'w'), indent=4)

        for i in range(s * m):
            judge_list = [data['generation'][i] for data in dev_res_list]
            score_list, parsed_output_list = eval.get_scores(task, task_type, "dev", judge_list, return_output=True)
            for data, parsed_output, score in zip(dev_res_list, parsed_output_list, score_list):
                if i == 0:
                    data['parsed_generation'] = [parsed_output]
                    data['score'] = [score]
                else:
                    data['parsed_generation'].append(parsed_output)
                    data['score'].append(score)

        def is_hard(generation_list, score_list):
            counter = Counter(generation_list)
            most_common = counter.most_common(1)[0][0]
            return score_list[generation_list.index(most_common)] != 1

        hard_dataset = []
        easy_dataset = []
        for data in dev_res_list:
            for i in range(s):
                if is_hard(data['parsed_generation'][i * m: i * m + m], data['score'][i * m: i * m + m]):
                    hard_dataset.append({
                        'id': data['id'],
                        'prompt': [{
                            'content': form_aggregator_instruction(data['input'], data['generation'][i * m: i * m + m]),
                            'role': 'user'
                        }]
                    })
                else:
                    easy_dataset.append({
                        'id': data['id'],
                        'prompt': [{
                            'content': form_aggregator_instruction(data['input'], data['generation'][i * m: i * m + m]),
                            'role': 'user'
                        }]
                    })
        random.seed(42)
        easy_dataset = random.sample(easy_dataset, min(len(easy_dataset), len(hard_dataset)))
        overall_dataset = hard_dataset + easy_dataset
        overall_dataset = Dataset.from_list(overall_dataset)

        gpu_id_str = ",".join([str(i) for i in gpu_ids])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_str

        training_args = GRPOConfig(
            num_generations=batch_size,
            gradient_accumulation_steps=1,
            per_device_train_batch_size=batch_size,
            do_eval=False,
            bf16=True,
            per_device_eval_batch_size=batch_size,
            save_strategy='steps',
            save_steps=200,
            logging_strategy='steps',
            logging_steps=200,
            save_total_limit=3,
            output_dir=agglm_log_path + '/' + file_name[:-5],
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler,
            weight_decay=weight_decay,
            num_train_epochs=max_epoches,
            warmup_ratio=0.1,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            seed=42,
            max_prompt_length=4096,
            max_completion_length=max_response_length,
            temperature=temperature,
        )
        peft_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            modules_to_save=None,
        )
        agg_model = AutoModelForCausalLM.from_pretrained(
            agg_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )

        def reward_func(prompts, completions, **kwargs):
            responses = [completion[0]['content'] for completion in completions]
            return eval.get_scores(task, task_type, "dev", responses, id_list=kwargs['id'])

        trainer = GRPOTrainer(
            model=agg_model,
            args=training_args,
            reward_funcs=reward_func,
            train_dataset=overall_dataset,
            peft_config=peft_config
        )
        trainer.train()
        if trainer.accelerator.is_main_process:
            trainer.save_model(agglm_log_path + '/' + file_name[:-5])
        trainer.accelerator.wait_for_everyone()

        del trainer
        del agg_model
        torch.cuda.empty_cache()
        _dynamo.reset_code_caches()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()

    new_gpu_ids = [i for i in range(len(model_names))]
    test_input_list = eval.prepare_inputs(task, task_type, 'test')
    list_of_test_output_list = distributed_generation.distributed_generation(
        model_names,
        [test_input_list for _ in model_names],
        new_gpu_ids,
    )
    agg_input_list = []
    for i in range(len(test_input_list)):
        test_output_list = [list_of_test_output_list[j][i] for j in range(m)]
        agg_input_list.append(form_aggregator_instruction(test_input_list[i], test_output_list))
    agg_output_list = distributed_generation.distributed_generation(
        [agglm_log_path + '/' + file_name[:-5]],
        [agg_input_list],
        [new_gpu_ids[0]]
    )
    test_scores = eval.get_scores(task, task_type, "test", agg_output_list[0])

    avg_test_score = sum(test_scores) / len(test_scores)
    print("Agglm test {} score: {}".format(task, avg_test_score))

    # save the logs
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_score,
        "logs": []
    }
    for i in range(len(test_input_list)):
        log = {
            "input": test_input_list[i],
            "output": agg_output_list[0][i],
            "score": test_scores[i]
        }
        experiment_logs["logs"].append(log)

    # file name with task, number of models, and avg_test_score with 4 decimal places
    log_filename = "model_collaboration/logs/{}_{}_{}_agglm.json".format(task, len(model_names), round(avg_test_score, 4))
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)

    return 0



if __name__ == "__main__":
    run_method()