# Import: Basic Python Libraries

import os
import torch

from torch.utils.data import DataLoader

# Import: Custom Python Libraries

from transforms import load_data_transforms
from loader import Dataset as Pytorch_Dataset

# Initialize: Standard Dataset

class Dataset:

    def __init__(self, samples, labels):

        self.labels = labels
        self.samples = samples

# Create: Dataset (Samples, Labels)

def gather_data(path, title, rank = 0, file_types = [".png"]):

    # Gather: All Dataset Files (Folders)

    all_folders = os.listdir(path)
    all_folders.sort()

    # Load: Supervised Learning Dataset

    if(rank == 0):
        print("\n-------------------------\n")

    all_samples, all_labels = [], []

    total_files = sum([len(os.listdir(os.path.join(path, folder))) for folder in all_folders])
    number = int(total_files * 0.20)

    count = 0
    for i, current_folder in enumerate(all_folders):

        all_files = os.listdir(os.path.join(path, current_folder))
        all_files.sort()

        for current_file in all_files:
            all_samples.append(os.path.join(path, current_folder, current_file))
            all_labels.append(i)
        
            if(rank == 0 and count % number == 0):
                print("%s : %s percent" % (title, round(count / total_files, 2) * 100))

            count = count + 1

    if(rank == 0):
        print("%s : 100 percent" % (title))

        if("Valid" in title or "Test" in title):
            print("\n-------------------------\n")

    return Dataset(all_samples, all_labels)

# Load: Datasets, (Train, Valid, Test)

def load_datasets_train(path_train, path_valid, path_test, rank = 0):

    train = gather_data(path_train, "Loading Train Data", rank)
    valid = gather_data(path_valid, "Loading Valid Data", rank)
    test = gather_data(path_test, "Loading Test Data", rank)

    return train, valid, test

# Load: Datasets, (Test Only)

def load_datasets_test(path_test, rank = 0):

    return gather_data(path_test, "Loading Test Data", rank)

# Create: DataLoaders, Training Pipeline

def run_train(params, rank):

    train, valid, test = load_datasets_train(params["path_train"], params["path_valid"], params["path_test"], rank)

    transforms = load_data_transforms(params["transforms"], params["interpolate"], params["sample_shape"])

    train = Pytorch_Dataset(train, transforms, params["sample_shape"][0], "train")
    valid = Pytorch_Dataset(valid, transforms, params["sample_shape"][0], "valid")
    test = Pytorch_Dataset(test, transforms, params["sample_shape"][0], "valid")

    train = DataLoader(train, batch_size = params["batch_size"], shuffle = 1)
    valid = DataLoader(valid, batch_size = params["batch_size"], shuffle = 1)
    test = DataLoader(test, batch_size = params["batch_size"], shuffle = 0)

    return train, valid, test

# Create: DataLoaders, Testing Pipeline

def run_test(params, rank):

    test = load_datasets_test(params["path_test"], rank)

    transforms = load_data_transforms(params["transforms"], params["interpolate"], params["sample_shape"])

    test = Pytorch_Dataset(test, transforms, params["sample_shape"][0], "valid")

    test = DataLoader(test, batch_size = params["batch_size"], shuffle = 0)

    return test
