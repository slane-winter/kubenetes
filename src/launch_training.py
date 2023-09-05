# Import: Standard Python Libraries

import os
import sys
import yaml
import time
import shutil
import atexit
import subprocess

from kubernetes import client, config
from jinja2 import Environment, FileSystemLoader

# Create: Response, Program Exit

def exit_handler(): 

    config.load_kube_config()

    v1 = client.CoreV1Api()

    pod_list = v1.list_namespaced_pod(namespace = params["resources"]["namespace"])

    current_group = [ele.metadata.owner_references[0].name for ele in pod_list.items if(params["helpers"]["kill_tag"] in ele.metadata.name)]

    current_group = list(set(current_group))

    for job_name in current_group:
        subprocess.run(["kubectl", "delete", "job", job_name])

    print("\nCleaned up any jobs that include tag : %s\n" % params["helpers"]["kill_tag"])   

# Create: Results Folders

def create_folder(path):

    if(os.path.exists(path)):
        shutil.rmtree(path)

    os.makedirs(path)

# Save: Template File

def save_file(path, data):

    data_file = open(path, "w")
   
    data_file.write(data) 

    data_file.close()

# Load: Template File

def load_file(path):

    data_file = open(path, "r")
    
    info = ""

    for line in data_file:
        info += line

    data_file.close()

    return info

# Run: Parallelized Physics Simulatiion

def run_generation(params):

    # Load: Job Template

    template = load_file(params["paths"]["template"])

    tag = params["paths"]["template"].split("/")[-1]
    folder = params["paths"]["template"].replace("/%s" % tag, "")
    environment = Environment(loader = FileSystemLoader(folder))
    template = environment.get_template(tag)

    # Launch: Algorithm Jobs

    create_folder(params["paths"]["job_files"])

    # - Begin: Algorithm Analysis

    print("\nLaunching Algorithm Jobs\n")

    num_exps = len(params["helpers"]["dataset_names"]) * len(params["helpers"]["algorithm_names"])

    counter = 0

    current_group = [] 

    for data_name in params["helpers"]["dataset_names"]:

        path_train = os.path.join(params["paths"]["root_data"], data_name, "train")
        path_valid = os.path.join(params["paths"]["root_data"], data_name, "valid")
        path_test = os.path.join(params["paths"]["root_data"], data_name, "test")

        for alg_name in params["helpers"]["algorithm_names"]:

            path_results = os.path.join(params["paths"]["root_results"], data_name, alg_name)

            if(sys.platform == "win32"):
                path_results = path_results.replace("\\", "/")
                path_train = path_train.replace("\\", "/")
                path_valid = path_valid.replace("\\", "/")
                path_test = path_test.replace("\\", "/")

            # -- Launch jobs if there is room in processing group

            if(len(current_group) < params["helpers"]["parallel_ops"]):

                num_to_launch = params["helpers"]["parallel_ops"] - len(current_group)

                for i in range(num_to_launch):

                    # --- Configure algorithm job

                    job_name = "%s-%s" % (params["helpers"]["kill_tag"], str(counter).zfill(6))
 
                    path_log = os.path.join(params["paths"]["root_logs"], "logs", "%s_%s_%s.log" % (job_name, data_name, alg_name))

                    if(sys.platform == "win32"):
                        path_log = path_log.replace("\\", "/") 

                    current_group.append(job_name)

                    template_info = {"job_name": job_name, 
                                     "mem_lim": params["resources"]["mem_lim"],
                                     "mem_req": params["resources"]["mem_req"],
                                     "num_cpus": str(params["resources"]["cpus_per_op"]),
                                     "num_gpus": str(params["resources"]["gpus_per_op"]),

                                     "log": path_log,
                                     "test": path_test,
                                     "valid": path_valid,
                                     "train": path_train,
                                     "results": path_results, 

                                     "arch": alg_name,

                                     "image": params["paths"]["container_image"]}

                    filled_template = template.render(template_info)

                    path_job = os.path.join(params["paths"]["job_files"], job_name + ".yaml") 

                    if(sys.platform == "win32"):
                        path_job = path_job.replace("\\", "/").replace("/", "\\")

                    # --- Save simulation job file

                    save_file(path_job, filled_template)

                    # --- Launch simulation job

                    subprocess.run(["kubectl", "apply", "-f", path_job])

                    counter += 1

            # -- Otherwise wait for a processes to finish

            else:

                k = 0
                check_time_min = 2
                wait_time_sec = 60

                while(len(current_group) == params["helpers"]["parallel_ops"]): 

                    time.sleep(wait_time_sec)

                    # --- Check progress every n minutes

                    if(k % check_time_min == 0): 

                        # --- Gather kubernetes information

                        config.load_kube_config()
                        v1 = client.CoreV1Api()
                        pod_list = v1.list_namespaced_pod(namespace = params["resources"]["namespace"], timeout_seconds = 300)
                
                        if(k == 0):
                            print()

                        pod_list = [item for item in pod_list.items if(params["helpers"]["kill_tag"] in item.metadata.name)]

                        pod_names = [item.metadata.name for item in pod_list]
                        pod_statuses = [item.status.phase for item in pod_list]

                        # --- Remove pods that have finished. Jobs and pods share the same name.
                        
                        pod_progress = [1 if(phase == "Succeeded" or phase == "Error") else 0 for phase in pod_statuses]

                        for i, (job_name, remove_flag) in enumerate(zip(current_group, pod_progress)):
                            if(remove_flag):
                                print()
                                #time.sleep(wait_time_sec)
                                subprocess.run(["kubectl", "delete", "job", job_name])
                                current_group.pop(i)
                                print()

                        print("Log: Elapsed Time = %s minutes, Group Size = %s, Total (In Progress) = %s / %s" % ((wait_time_sec * (k + 1)) / 60, len(current_group), counter, num_exps))

                        if(sum(pod_progress) > 0):
                            print("\nJobs Finished. Updating...\n")
                            break                    

                    k += 1

    print("\nData Generation Complete\n")

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

    atexit.register(exit_handler)  # this is how we clean up jobs. 

    run_generation(params)
