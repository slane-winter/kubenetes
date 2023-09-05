# Initialize: All System Paths

import os
import sys
import torch

sys.path.append("network")
sys.path.append("loading")

# Import: Standard Python Libraries

import yaml

# Import: Custom Python Libraries

import prepare_model 

# Update: Parameters, User Flags

def update_params(params, args):

    # Update: Command-line Arguments Override

    arg_list = list(args.keys())

    if("results" in arg_list):
        params["paths"]["results"] = args["results"]

    if("train" in arg_list):
        params["paths"]["train"] = args["train"]

    if("valid" in arg_list):
        params["paths"]["valid"] = args["valid"]

    if("test" in arg_list):
        params["paths"]["test"] = args["test"]

    if("deploy" in arg_list):
        params["network"]["deploy"] = args["deploy"] 

    if("arch" in arg_list):
        params["network"]["arch"] = int(args["arch"])

    if("use_gpu" in arg_list):
        params["system"]["gpu_config"]["use_gpu"] = args["use_gpu"]

    # Update: System Config

    if(params["system"]["gpu_config"]["use_gpu"]):
        name = "CUDA"
        total_num_gpus = os.environ["WORLD_SIZE"]
        num_gpus_per_node = torch.cuda.device_count() 
        num_nodes = int(total_num_gpus) // int(num_gpus_per_node)
    else:
        name = "CPU"
        num_nodes = 1
        total_num_gpus = 0
        num_gpus_per_node = 0

    params["system"]["gpu_config"]["device"] = name
    params["system"]["gpu_config"]["platform"] = sys.platform
    params["system"]["gpu_config"]["num_nodes"] = num_nodes
    params["system"]["gpu_config"]["total_num_gpus"] = total_num_gpus
    params["system"]["gpu_config"]["num_gpus_per_node"] = num_gpus_per_node

    # Update: Dataset Config

    params["dataset"]["path_train"] = params["paths"]["train"]
    params["dataset"]["path_valid"] = params["paths"]["valid"]
    params["dataset"]["path_test"] = params["paths"]["test"]

    # Update: Paths Config

    if(os.path.exists(params["paths"]["results"])):
        content = [ele for ele in os.listdir(params["paths"]["results"]) if(".checkpoint" in ele)]
        content.sort()
    else:
        content = []

    if(len(content) != 0 and params["network"]["use_progress"]):
        params["paths"]["network"] = os.path.join(params["paths"]["results"], content[-1])
        params["network"]["path_network"] = params["paths"]["network"]
    else:
        params["network"]["path_network"] = ""

    params["network"]["sample_shape"] = params["dataset"]["sample_shape"]
    params["network"]["path_results"] = params["paths"]["results"]

    # Update: Network Config

    params["network"]["num_classes"] = params["dataset"]["num_classes"]

# Validate: Configuration File

def load_config(argument):

    try:
        return yaml.load(open(argument), Loader = yaml.FullLoader) 

    except Exception as e:
        print("\nError: Loading YAML Configuration File") 
        print("\nSuggestion: Using YAML file? Check File For Errors\n")
        print(e)
        exit()        

# Parse: Command-line Arguments

def parse_args(all_args, tags = ["--", "-"]):

    all_args = all_args[1:]

    if(len(all_args) % 2 != 0):
        print("Argument '%s' not defined" % all_args[-1])
        exit()

    results = {}

    i = 0
    while(i < len(all_args) - 1):
        arg = all_args[i].lower()
        for current_tag in tags:
            if(current_tag in arg):
                arg = arg.replace(current_tag, "")                
        results[arg] = all_args[i + 1]
        i += 2

    return results

# Main: Load Configuration File

if __name__ == "__main__":
    
    args = parse_args(sys.argv)

    params = load_config(args["config"]) 

    update_params(params, args)

    prepare_model.run(params)
