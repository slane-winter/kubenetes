# Import: Standard Python Libraries

import os
import torch

from torch.nn.parallel import DistributedDataParallel as DDP

# Import: Custom Python Libraries

from model import Classifier

# Saving: Model Training Progress

def save_progress(model, path, count = 4):

    if(hasattr(model, "module")):
        start = model.module.current_epoch
    else:
        start = model.current_epoch

    [os.remove(os.path.join(path, ele)) for ele in os.listdir(path) if(".checkpoint" in ele)]
    name = str(start).zfill(count) + ".checkpoint"

    try:
        torch.save(model.state_dict(), os.path.join(path, name))
    except KeyboardInterrupt:
        torch.save(model.state_dict(), os.path.join(path, name))
        exit()

# Loading: Model Training Progress

def load_progress(model, path, gpu_config, rank, tag = ".checkpoint"):

    if(os.path.exists(path)):
        if(gpu_config["platform"] == "darwin"):
            history = torch.load(path)
        else:
            if(gpu_config["use_gpu"]):
                map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
                history = torch.load(path, map_location = map_location)
            else:    
                history = torch.load(path)

        filename = path.split("/")[-1]

        model.load_state_dict(history)

        if(hasattr(model, "module")):
            model.module.current_epoch = int(filename.strip(tag))
        else:
            model.current_epoch = int(filename.strip(tag))

    return model

# Initialize: GPU Configuration

def config_hardware(model, gpu_config, rank = 0):

    if(gpu_config["use_gpu"]):

        # Config: M1-MacOS GPU

        if(gpu_config["platform"] == "darwin"):
            model = model.to("mps")
        
        # Config: NVIDIA GPU(s)

        else:

            torch.distributed.init_process_group("nccl")
            rank = torch.distributed.get_rank()             

            device_id = rank % torch.cuda.device_count()
            model = model.to(device_id)
            model = DDP(model, device_ids=[device_id])

    return model, rank

# Initialize: Neural Network

def select_network(network_params, system_params):

    # Load: Network

    model = Classifier(network_params)

    model, rank = config_hardware(model, system_params["gpu_config"])
    
    # Selection: Start Over || Continue

    if(network_params["use_progress"] and rank == 0):
        model = load_progress(model, network_params["path_network"], system_params["gpu_config"], rank)

    network_params["rank"] = rank

    return model
