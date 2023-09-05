# Import: Standard Python Libraries

import torch

# Import: Custom Python Libraries

from prepare_data import run_train, run_test
from model_utils import select_network, save_progress
from logger import log_exp, log_results, log_predict, create_results

# Create: Testing Procedure

def testing(model, datasets, params):

    if(hasattr(model, "module")):
        cycle = model.module.test_cycle
    else:
        cycle = model.test_cycle

    if(isinstance(datasets, list)):
        _, _, test_data = datasets
    else:
        test_data = datasets

    model.eval() 

    results = cycle(test_data)

    if(params["rank"] == 0):
        log_predict(results, params)
    
# Create: Training Procedure

def train_and_validate(model, datasets, params):

    if(hasattr(model, "module")):
        start = model.module.current_epoch
        cycle = model.module.epoch_cycle
    else:
        start = model.current_epoch
        cycle = model.epoch_cycle

    train_data, valid_data, _ = datasets

    for epoch in range(start, params["num_epochs"]):
       
        # Train: Network
 
        results = cycle(train_data, "train")

        if(params["rank"] == 0):
            log_results(results, epoch, params, "train")
            save_progress(model, params["path_results"])

        # Valid: Network
    
        if(epoch % params["valid_rate"] == 0):

            model.eval() 

            results = cycle(valid_data, "valid")
   
            if(params["rank"] == 0):
                log_results(results, epoch, params, "valid")

            model.train()


# Set: Experiment Reproducability
# - Seeding sets deterministic randoms

def seed_everything(params):

    if(params["seed"]["flag"]):
        torch.manual_seed(params["seed"]["sequence"])
        torch.cuda.manual_seed(params["seed"]["sequence"])
        
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

# Create: Algorithm Pipeline

def run(params):

    # Set: Global Randomization Seed

    seed_everything(params["system"])

    # Selection: Testing Pipeline

    if(int(params["network"]["deploy"])):

        # - Load pre-trained model

        params["network"]["use_progress"] = 1
        model = select_network(params["network"], params["system"])

        # - Display experiment configuration

        if(params["network"]["rank"] == 0): 
            print("\nTesting Experiment\n")
            params["paths"]["train"] = None
            params["paths"]["valid"] = None
            log_exp(params["paths"], params["system"]["gpu_config"])

        # - Load testing dataset

        test = run_test(params["dataset"], params["network"]["rank"])

        # - Run testing procedure
   
        testing(model, test, params["network"]) 

        if(params["network"]["rank"] == 0): 
            print("\nTesting Procedure Complete\n")

    else:

        # - Load pre-trained model

        params["network"]["use_progress"] = 1
        model = select_network(params["network"], params["system"])

        # - Display experiment configuration

        if(params["network"]["rank"] == 0): 
            print("\nTraining Experiment\n")

            log_exp(params["paths"], params["system"]["gpu_config"])
            create_results(params["network"])

            # -- Create results folder

        # - Load training, validation, and testing datasets

        datasets = run_train(params["dataset"], params["network"]["rank"])

        # - Run training procedure

        train_and_validate(model, datasets, params["network"])

        # - Run testing procedure
   
        testing(model, datasets, params["network"]) 

        if(params["network"]["rank"] == 0): 
            print("\nTraining Procedure Complete\n")

