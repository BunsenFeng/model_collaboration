import os
import math
import json
import torch
import shutil
import random
from multiprocessing import Pool
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft.utils.save_and_load import set_peft_model_state_dict, get_peft_model_state_dict

def uneven_fuse(weights, tensors):
    # fuse a list of tensors with uneven sizes
    # extend to the same dimension with zeros, and only weighted average on the overlapping parts
    # the list of tensors will be (?, n), with only the first dimension uneven

    # if the size is even, just do normal weighted average
    if all([tensor.size(0) == tensors[0].size(0) for tensor in tensors]):
        fused_tensor = torch.zeros_like(tensors[0])
        for i in range(len(tensors)):
            fused_tensor += weights[i] * tensors[i]
        return fused_tensor

    assert len(weights) == len(tensors)
    max_size = max([tensor.size(0) for tensor in tensors])
    fused_tensor = torch.zeros((max_size, *tensors[0].size()[1:]))
    weight_sums = torch.zeros((max_size,))
    for i in range(len(tensors)):
        tensor = tensors[i]
        weight = weights[i]
        size = tensor.size(0)
        fused_tensor[:size] += weight * tensor
        weight_sums[:size] += weight
    # avoid division by zero
    weight_sums[weight_sums == 0] = 1.0
    fused_tensor = fused_tensor / weight_sums.unsqueeze(-1)
    return fused_tensor

def lora_merge(weights, lora_name_list, output_path, gpu_id, directly_load_safetensors = 0, base_model=None):

    output_name = output_path

    # the slow merge
    if not directly_load_safetensors:
        # mergekit implementation
        with open("logs/mergekit_args.yml", "w") as f:
            f.write("models:\n")
            for i in range(len(lora_name_list)):
                f.write("  - model: " + lora_name_list[i] + "\n")
                f.write("    parameters:\n")
                f.write("      weight: " + str(weights[i]) + "\n")
            f.write("merge_method: linear\n")
            f.write("dtype: float16\n")
        
        # executing it
        os.system("mergekit-yaml logs/mergekit_args.yml " + output_name + " --cuda --device cuda:" + str(gpu_id))

        # lora_state_dict_list = []
        # for lora_name in lora_name_list:
        #     model = AutoModelForCausalLM.from_pretrained(lora_name)
        #     # lora_state_dict_list.append(get_peft_model_state_dict(model))
        #     lora_state_dict_list.append(model.state_dict())
        #     if not lora_name == lora_name_list[-1]:
        #         del model
        #     # torch.cuda.empty_cache()
        
        # final_state_dict = {}

        # # for key in lora_state_dict_list[0].keys():
        # #     for i in range(len(lora_state_dict_list)):
        # #         assert key in lora_state_dict_list[i].keys()
        # #     final_state_dict[key] = uneven_fuse(weights, [lora_state_dict_list[i][key] for i in range(len(lora_state_dict_list))])

        # # multiprocessing of uneven_fuse over keys
        # uneven_fuse_args = []
        # for key in lora_state_dict_list[0].keys():
        #     for i in range(len(lora_state_dict_list)):
        #         assert key in lora_state_dict_list[i].keys()
        #     uneven_fuse_args.append((weights, [lora_state_dict_list[i][key] for i in range(len(lora_state_dict_list))]))
        
        # # with Pool(processes=8) as p:
        # #     uneven_fuse_results = p.starmap(uneven_fuse, uneven_fuse_args)
        
        # # non-multiprocessing version
        # uneven_fuse_results = []
        # for args in uneven_fuse_args:
        #     uneven_fuse_results.append(uneven_fuse(*args))
        
        # for i, key in enumerate(lora_state_dict_list[0].keys()):
        #     final_state_dict[key] = uneven_fuse_results[i]

        # # for i in range(len(lora_state_dict_list)):
        # #     if i == 0:
        # #         for key in lora_state_dict_list[i].keys():
        # #             final_state_dict[key] = weights[i] * lora_state_dict_list[i][key]
        # #     else:
        # #         for key in lora_state_dict_list[i].keys():
        # #             assert key in final_state_dict.keys()
        # #             final_state_dict[key] += weights[i] * lora_state_dict_list[i][key]
        
        # # model = AutoModelForCausalLM.from_pretrained(lora_name_list[0]).to(f"cuda:{gpu_id}")
        # # set_peft_model_state_dict(model, final_state_dict)
        # # set model state dict directly
        # try:
        #     model = AutoModelForCausalLM.from_pretrained(base_model)
        #     model.load_state_dict(final_state_dict, strict=False)
        # except:
        #     for i in range(len(lora_name_list)):
        #         try:
        #             model = AutoModelForCausalLM.from_pretrained(lora_name_list[i])
        #             model.load_state_dict(final_state_dict, strict=False)
        #             break
        #         except:
        #             continue
        # if os.path.exists(output_name):
        #     shutil.rmtree(output_name)
        # model.save_pretrained(output_name)
    else:
        # the fast merge: load only state_dicts, merge them, save only state_dicts, gpu_id not used here
        # apply to the setting that models share the same architecture, sharding, and adapter format
        lora_state_dict_list = []
        for lora_name in lora_name_list:
            state_dict_this = load_file(os.path.join(lora_name, "adapter_model.safetensors"), device="cpu")
            lora_state_dict_list.append(state_dict_this)
        
        final_state_dict = {}
        for i in range(len(lora_state_dict_list)):
            if i == 0:
                for key in lora_state_dict_list[i].keys():
                    final_state_dict[key] = weights[i] * lora_state_dict_list[i][key]
            else:
                for key in lora_state_dict_list[i].keys():
                    assert key in final_state_dict.keys()
                    final_state_dict[key] += weights[i] * lora_state_dict_list[i][key]
        
        if not os.path.exists(output_name):
            os.mkdir(output_name)
        save_file(final_state_dict, os.path.join(output_name, "adapter_model.safetensors"))

        return final_state_dict

# sanity check example
# lora_merge([0.3, 0.6, 0.8], ["./initial_experts/lima", "./initial_experts/cot", "./initial_experts/oasst1"], "./new", 0, directly_load_safetensors=1)

# define the swarm class
# managing initialization and update of the swarm

def assign_gpu(num_gpus, process_idx, total_processes):
    process_per_gpu = math.ceil(total_processes / num_gpus)
    gpu_idx = math.floor(process_idx / process_per_gpu)
    return gpu_idx

class Swarm:

    def __init__(self, swarm_base_path, model_paths, base_model, fast_merge, starting_velocity_mode, weight_randomness, inertia, cognitive_coeff, social_coeff, repel_coeff, step_length, repel_term, step_length_factor, minimum_step_length, gpus, patience, restart_patience):

        self.swarm_base_path = swarm_base_path
        if not os.path.exists(self.swarm_base_path):
            os.mkdir(self.swarm_base_path)
        else:
            # remove and recreate
            shutil.rmtree(self.swarm_base_path)
            os.mkdir(self.swarm_base_path)
        self.model_paths = model_paths
        self.base_model = base_model
        self.fast_merge = fast_merge
        self.starting_velocity_mode = starting_velocity_mode
        self.weight_randomness = weight_randomness
        self.inertia = inertia
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.repel_coeff = repel_coeff
        self.step_length = step_length
        self.repel_term = repel_term
        self.step_length_factor = step_length_factor
        self.minimum_step_length = minimum_step_length
        self.gpus = gpus
        self.patience = patience
        self.restart_patience = restart_patience
        self.restart_counter = [0] * len(self.model_paths)

        # initialize utility scratchpad
        self.utility_scratchpad = {"g": None, "g_worst": None, "g_history": []}
        for i in range(len(self.model_paths)):
            self.utility_scratchpad[f"model_{i}_now"] = None
            self.utility_scratchpad[f"model_{i}_best"] = None
            self.utility_scratchpad[f"model_{i}_history"] = []
        
        with open(os.path.join(swarm_base_path, "utility_scratchpad.json"), "w") as f:
            json.dump(self.utility_scratchpad, f, indent=4)

        # iniitalize the directories for the swarm
        for i in range(len(self.model_paths)):
            os.mkdir(os.path.join(self.swarm_base_path, "model_" + str(i)))
            for checkpoint_type in ["personal_best", "now", "velocity"]:
                os.mkdir(os.path.join(self.swarm_base_path, "model_" + str(i), checkpoint_type))
        os.mkdir(os.path.join(self.swarm_base_path, "global_best"))
        os.mkdir(os.path.join(self.swarm_base_path, "global_worst"))

        # initialize model now weights and personal_best
        for i in range(len(self.model_paths)):
            try: # if it is an existing local directory
                shutil.copytree(self.model_paths[i], os.path.join(self.swarm_base_path, "model_" + str(i), "now"), dirs_exist_ok=True)
            except: # else, assume it is a huggingface repo id, download, and save to the local directory
                
                local_model_path = os.path.join(self.swarm_base_path, "temp_model_" + str(i))
                snapshot_download(repo_id=self.model_paths[i], local_dir=local_model_path)
                shutil.copytree(local_model_path, os.path.join(self.swarm_base_path, "model_" + str(i), "now"), dirs_exist_ok=True)
                shutil.rmtree(local_model_path)

            shutil.copytree(os.path.join(self.swarm_base_path, "model_" + str(i), "now"), os.path.join(self.swarm_base_path, "model_" + str(i), "personal_best"), dirs_exist_ok=True)

        # initialize model velocity
        if starting_velocity_mode == "random":
            merge_args = []
            for i in range(len(self.model_paths)):
                secret_lover_id = random.randint(0, len(self.model_paths) - 1)
                while secret_lover_id == i:
                    secret_lover_id = random.randint(0, len(self.model_paths) - 1)
                merge_args.append(([-1,1], [os.path.join(self.swarm_base_path, "model_" + str(i), "now"), os.path.join(self.swarm_base_path, "model_" + str(secret_lover_id), "now")], os.path.join(self.swarm_base_path, "model_" + str(i), "velocity"), gpus[assign_gpu(len(gpus), i, len(self.model_paths))], fast_merge, self.base_model))

            # with Pool(len(gpus)) as p:
            #     p.starmap(lora_merge, merge_args, chunksize = math.ceil(len(model_paths)/len(gpus)))

            # non-multiprocessing version
            for args in merge_args:
                lora_merge(*args)

        else:
            raise NotImplementedError

        # no starting utility eval: will eval then update in iteration 1

        # initialize global best and global worst
        for checkpoint_type in ["global_best", "global_worst"]:
            random_idx = random.randint(0, len(self.model_paths) - 1)
            shutil.copytree(os.path.join(self.swarm_base_path, "model_" + str(random_idx), "now"), os.path.join(self.swarm_base_path, checkpoint_type), dirs_exist_ok=True)

    def update(self, scores):
        assert len(scores) == len(self.model_paths)

        # update utility scratchpad
        for i in range(len(self.model_paths)):
            self.utility_scratchpad[f"model_{i}_now"] = scores[i]
            self.utility_scratchpad[f"model_{i}_history"].append(scores[i])
            if self.utility_scratchpad[f"model_{i}_best"] is None or scores[i] > self.utility_scratchpad[f"model_{i}_best"]:
                self.utility_scratchpad[f"model_{i}_best"] = scores[i]
                shutil.copytree(os.path.join(self.swarm_base_path, "model_" + str(i), "now"), os.path.join(self.swarm_base_path, "model_" + str(i), "personal_best"), dirs_exist_ok=True)
        
        if self.utility_scratchpad["g"] is None or max(scores) > self.utility_scratchpad["g"]:
            self.utility_scratchpad["g"] = max(scores)
            best_idx = scores.index(max(scores))
            shutil.copytree(os.path.join(self.swarm_base_path, "model_" + str(best_idx), "now"), os.path.join(self.swarm_base_path, "global_best"), dirs_exist_ok=True)
        if self.utility_scratchpad["g_worst"] is None or min(scores) < self.utility_scratchpad["g_worst"]:
            self.utility_scratchpad["g_worst"] = min(scores)
            worst_idx = scores.index(min(scores))
            shutil.copytree(os.path.join(self.swarm_base_path, "model_" + str(worst_idx), "now"), os.path.join(self.swarm_base_path, "global_worst"), dirs_exist_ok=True)
        self.utility_scratchpad["g_history"].append(self.utility_scratchpad["g"])

        with open(os.path.join(self.swarm_base_path, "utility_scratchpad.json"), "w") as f:
            json.dump(self.utility_scratchpad, f, indent=4)

        # if "g_history" did not improve in patience iterations, terminate signal
        if len(self.utility_scratchpad["g_history"]) > self.patience and self.utility_scratchpad["g_history"][-1] <= self.utility_scratchpad["g_history"][-self.patience]:
            termination_flag = True
            return termination_flag
        
        for i in range(len(self.model_paths)):
            
            base_path = self.swarm_base_path
            model_path = os.path.join(self.swarm_base_path, "model_" + str(i))
            now_path = os.path.join(model_path, "now")
            best_path = os.path.join(model_path, "personal_best")
            velocity_path = os.path.join(model_path, "velocity")

            # judge restart flag
            if len(self.utility_scratchpad[f"model_{i}_history"]) > self.restart_patience and self.utility_scratchpad[f"model_{i}_history"][-1] <= self.utility_scratchpad[f"model_{i}_history"][-int(self.restart_patience)] and self.restart_counter[i] == 0:
                restart_flag = True
                self.restart_counter[i] = 3
            else:
                restart_flag = False
                self.restart_counter[i] = max(0, self.restart_counter[i] - 1)

            if restart_flag:
                shutil.copytree(best_path, now_path, dirs_exist_ok=True)
                lora_merge([0], [now_path], velocity_path, self.gpus[assign_gpu(len(self.gpus), i, len(self.model_paths))], self.fast_merge, self.base_model)
            
            # weight randomness
            if self.weight_randomness:
                r_w = random.uniform(0, 1)
                r_p = random.uniform(0, 1)
                r_s = random.uniform(0, 1)
                r_b = random.uniform(0, 1) # b for bad, repel term weight
            else:
                r_w = 1
                r_p = 1
                r_s = 1
                r_b = 1
            
            # weight normalize
            self_weight = r_w * self.inertia
            cognitive_weight = r_p * self.cognitive_coeff
            social_weight = r_s * self.social_coeff
            repel_weight = r_b * self.repel_coeff if self.repel_term else 0
            weight_sum = self_weight + cognitive_weight + social_weight + repel_weight

            self_weight /= weight_sum
            cognitive_weight /= weight_sum
            social_weight /= weight_sum
            repel_weight /= weight_sum

            # unified one-step velocity update
            # v' = self_weight * v + cognitive_weight * (p_i - x_i) + social_weight * (g - x_i) + repel_weight * (x_i - w)
            #    = self_weight * v + cognitive_weight * p_i + social_weight * g + repel_weight * (-w) + (-cognitive_weight - social_weight + repel_weight) * x_i

            lora_merge(
                weights = [self_weight, cognitive_weight, social_weight, -repel_weight, -cognitive_weight - social_weight + repel_weight],
                lora_name_list = [
                    os.path.join(model_path, "velocity"),
                    os.path.join(model_path, "personal_best"),
                    os.path.join(self.swarm_base_path, "global_best"),
                    os.path.join(self.swarm_base_path, "global_worst"),
                    os.path.join(model_path, "now")
                ],
                output_path = os.path.join(model_path, "velocity"),
                gpu_id = self.gpus[assign_gpu(len(self.gpus), i, len(self.model_paths))],
                directly_load_safetensors = self.fast_merge,
                base_model = self.base_model
            )

            # # p_i-x_i task vector
            # lora_merge(
            #     weights = [1, -1],
            #     lora_name_list = [os.path.join(model_path, "personal_best"), os.path.join(model_path, "now")],
            #     output_path = os.path.join(model_path, "p_x"),
            #     gpu_id = self.gpus[assign_gpu(len(self.gpus), i, len(self.model_paths))],
            #     directly_load_safetensors = self.fast_merge,
            #     base_model = self.base_model
            # )

            # # g-x_i task vector
            # lora_merge(
            #     weights = [1, -1],
            #     lora_name_list = [os.path.join(self.swarm_base_path, "global_best"), os.path.join(model_path, "now")],
            #     output_path = os.path.join(model_path, "g_x"),
            #     gpu_id = self.gpus[assign_gpu(len(self.gpus), i, len(self.model_paths))],
            #     directly_load_safetensors = self.fast_merge,
            #     base_model = self.base_model
            # )

            # # x_i-w task vector
            # lora_merge(
            #     weights = [-1, 1],
            #     lora_name_list = [os.path.join(self.swarm_base_path, "global_worst"), os.path.join(model_path, "now")],
            #     output_path = os.path.join(model_path, "x_w"),
            #     gpu_id = self.gpus[assign_gpu(len(self.gpus), i, len(self.model_paths))],
            #     directly_load_safetensors = self.fast_merge,
            #     base_model = self.base_model
            # )

            # # update velocity
            # lora_merge(
            #     weights = [self_weight, cognitive_weight, social_weight, repel_weight],
            #     lora_name_list = [
            #         os.path.join(model_path, "velocity"),
            #         os.path.join(model_path, "p_x"),
            #         os.path.join(model_path, "g_x"),
            #         os.path.join(model_path, "x_w")
            #     ],
            #     output_path = os.path.join(model_path, "velocity"),
            #     gpu_id = self.gpus[assign_gpu(len(self.gpus), i, len(self.model_paths))],
            #     directly_load_safetensors = self.fast_merge,
            #     base_model = self.base_model
            # )

            # update now, current position
            lora_merge(
                weights = [1, self.step_length],
                lora_name_list = [
                    os.path.join(model_path, "now"),
                    os.path.join(model_path, "velocity")
                ],
                output_path = os.path.join(model_path, "now"),
                gpu_id = self.gpus[assign_gpu(len(self.gpus), i, len(self.model_paths))],
                directly_load_safetensors = self.fast_merge,
                base_model = self.base_model
            )

        # update step length
        self.step_length *= self.step_length_factor
        self.step_length = max(self.step_length, self.minimum_step_length)

        termination_flag = False
        return termination_flag

    def get_model_paths(self):
        now_paths = []
        for i in range(len(self.model_paths)):
            now_paths.append(os.path.join(self.swarm_base_path, "model_" + str(i), "now"))
        return now_paths

    def get_global_best_path(self):
        return os.path.join(self.swarm_base_path, "global_best")

    def clean_up(self):
        # remove everything except the global best folder and the utility scratchpad json file
        for item in os.listdir(self.swarm_base_path):
            item_path = os.path.join(self.swarm_base_path, item)
            if item == "global_best" or item == "utility_scratchpad.json":
                continue
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

if __name__ == '__main__':
    # demo: lora model

    # swarm = Swarm(
    #     swarm_base_path = "logs/model_swarms",
    #     model_paths = ["initial_experts/lima", "initial_experts/cot", "initial_experts/oasst1", "initial_experts/science"],
    #     base_model = "google/gemma-2-9b-it", # should be the model with the largest dict (b.c. special tokens etc.)
    #     fast_merge = True,
    #     starting_velocity_mode = "random",
    #     weight_randomness = True,
    #     inertia = 0.5,
    #     cognitive_coeff = 0.5,
    #     social_coeff = 0.5,
    #     repel_coeff = 0.5,
    #     step_length = 0.5,
    #     repel_term = True,
    #     step_length_factor = 0.95,
    #     minimum_step_length = 0.1,
    #     gpus = [0,1,2,3],
    #     patience = 5,
    #     restart_patience = 3
    # )

    # # flag = swarm.update([0.5, 0.6, 0.7, 0.8])
    # # flag = swarm.update([0.6, 0.7, 0.8, 0.8])
    # flag = swarm.update([0.7, 0.8, 0.9, 0.8])

    # print(swarm.get_model_paths())
    # print(swarm.get_global_best_path())

    # model = AutoModelForCausalLM.from_pretrained(swarm.get_global_best_path(), torch_dtype=torch.bfloat16).to("cuda:0")
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    # prompt = "Explain the theory of relativity in simple terms."
    # inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    # outputs = model.generate(
    #     **inputs,
    #     max_new_tokens=100,
    #     temperature=0.7,
    #     top_p=0.9,
    #     do_sample=True,
    #     pad_token_id=tokenizer.eos_token_id
    # )
    # output = tokenizer.decode(outputs[:, inputs.input_ids.shape[1]:][0], skip_special_tokens=True)
    # print(output)

    # demo: full model

    swarm = Swarm(
        swarm_base_path = "logs/model_swarms",
        model_paths = ["logs/model_0/now", "logs/model_1/now", "logs/model_2/now"],
        base_model = "logs/model_2/now", # should be the model with the largest dict (b.c. special tokens etc.)
        fast_merge = False,
        starting_velocity_mode = "random",
        weight_randomness = True,
        inertia = 0.5,
        cognitive_coeff = 0.5,
        social_coeff = 0.5,
        repel_coeff = 0.5,
        step_length = 0.5,
        repel_term = True,
        step_length_factor = 0.95,
        minimum_step_length = 0.1,
        gpus = [0,1,2],
        patience = 5,
        restart_patience = 3
    )

    flag = swarm.update([0.5, 0.6, 0.7])
    flag = swarm.update([0.6, 0.7, 0.8])
    # flag = swarm.update([0.7, 0.8, 0.9])

    print(swarm.get_model_paths())
    print(swarm.get_global_best_path())

    model = AutoModelForCausalLM.from_pretrained(swarm.get_global_best_path(), torch_dtype=torch.bfloat16).to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(swarm.get_global_best_path())
    prompt = "Explain the theory of relativity in simple terms."
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    output = tokenizer.decode(outputs[:, inputs.input_ids.shape[1]:][0], skip_special_tokens=True)
    print(output)

    # multiprocessing dummy demo
    # dummy_args = [(1,2), (3,4), (5,6), (7,8)]
    # with Pool(processes=4) as p:
    #     dummy_results = p.starmap(dummy, dummy_args)
    # print(dummy_results)

    # multiprocessing uneven_fuse demo

    # uneven_fuse_args = [
    #     ([0.2, 0.3, 0.5], [torch.randn(5,3), torch.randn(7,3), torch.randn(6,3)]),
    #     ([0.5, 0.5], [torch.randn(4,2), torch.randn(4,2)]),
    #     ([0.4, 0.6], [torch.randn(3,2), torch.randn(5,2)])
    # ]

    # # with Pool(processes=4) as p:
    # #     uneven_fuse_results = p.starmap(uneven_fuse, uneven_fuse_args)
    # pool = Pool(processes=4)
    # uneven_fuse_results = pool.starmap(uneven_fuse, uneven_fuse_args)
    # pool.close()
    # pool.join()

    # print(uneven_fuse_results)

    # print(uneven_fuse_results)

    # lora_merge demo
    # weights = [0.2, 0.3, 0.5]
    # lora_name_list = [
    #     "meta-llama/Llama-3.1-8B",
    #     "allenai/Llama-3.1-Tulu-3-8B-SFT",
    #     "allenai/Llama-3.1-Tulu-3-8B"
    # ]
    # base_model = "allenai/Llama-3.1-Tulu-3-8B"
    # output_path = "logs/merged_model"
    # gpu_id = 3
    # lora_merge(weights, lora_name_list, output_path, gpu_id, directly_load_safetensors=0, base_model=base_model)