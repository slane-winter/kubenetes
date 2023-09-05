# Import: Standard Python Libraries

import os
import shutil
import pickle
import pandas as pd

# Tracking: Testing Predictions

def log_predict(results, params):

    path_file = os.path.join(params["path_results"], "test_preds.pkl")

    with open(path_file, "wb") as handle:
        pickle.dump(results, handle, protocol = pickle.HIGHEST_PROTOCOL)

# Tracking: Training Performance 

def log_results(results, epoch, params, title):
 
    # Display: Current Progress

    if(title == "valid"):
        print("\n-------------------------\n")

    results["epoch"] = epoch
    
    print("%s results :" % (title), end = " ")
    for i, current_key in enumerate(results.keys()):
        end = "\n" if(i == len(results.keys()) - 1) else " "

        start = "| " if(current_key == "epoch") else ""
        print(start + "%s = %s" % (current_key, round(results[current_key], 4)), end = end)

    # Update: Total Progress 

    path_file = os.path.join(params["path_results"], title + "_loss.csv")

    results = pd.DataFrame([results])

    if(not(os.path.exists(path_file))):
        results.to_csv(path_file, index = False)
    else:
        data_file = pd.read_csv(path_file)
        data_file = pd.concat((data_file, results))
        data_file.to_csv(path_file, index = False)

    if(title == "valid"):
        print("\n-------------------------\n")

# Visualize: Experiment Details

def log_exp(path_params, system_params):

    print("\n-------------------------\n")

    print("Experiment: System\n")
    for current_key in system_params.keys():
        print("%s : %s" % (current_key, system_params[current_key]))

    print("\n-------------------------\n")

    print("Experiment: Paths\n")
    for current_key in path_params.keys():
        print("%s : %s" % (current_key, path_params[current_key]))

# Initialize: Folder, Exp Results

def create_results(params):

    if(not(os.path.exists(params["path_results"]))):
        os.makedirs(params["path_results"])
    else:
        if(not(params["use_progress"])):
            shutil.rmtree(params["path_results"])
            os.makedirs(params["path_results"])

