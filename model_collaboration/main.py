import os
import sys
import json
import torch
import argparse
import importlib
import torch._dynamo as dynamo
from data import eval
from multiprocessing import Pool
from method import distributed_generation

def run_main():
    torch.multiprocessing.set_start_method('spawn')

    torch.set_float32_matmul_precision('high')
    dynamo.config.cache_size_limit = 1024
    dynamo.config.recompile_limit = 1024

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = json.load(f)

    method_name = config["method"]
    task = config["task"]
    task_type = config["task_type"]
    gpu_ids = config["gpu_ids"]
    model_names = config["model_names"]
    hyperparameters = config["hyperparameters"]

    distributed_generation.update_generation_hyperparameters(
        hyperparameters.get("max_response_length", 512),
        hyperparameters.get("temperature", 0.7),
        hyperparameters.get("top_p", 0.9),
        hyperparameters.get("batch_size", 32),
        hyperparameters.get("big_model_mode", False)
    )

    # execute the method
    module_path = f"method.{method_name}"

    # try:
    print(f"Attempting to load module: {module_path}")
    method_module = importlib.import_module(module_path)

    if hasattr(method_module, 'run_method'):
        result = method_module.run_method(
            task, task_type, gpu_ids, model_names, hyperparameters
        )
        print(f"Method '{method_name}' executed successfully")
    else:
        raise AttributeError(f"The module '{module_path}' does not have a 'run_method' function.")
    # except ImportError as e:
    #     print(f"Error importing module '{module_path}': {e}")
    #     sys.exit(1)
    # except Exception as e:
    #     print(f"An error occurred while executing the method: {e}")
    #     sys.exit(1)

if __name__ == "__main__":
    run_main()
