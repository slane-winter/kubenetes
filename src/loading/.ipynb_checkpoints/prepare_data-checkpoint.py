#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import os
import numpy as np

from torch.utils.data import DataLoader

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

from transforms import load_data_transforms
from loader import Dataset as Pytorch_Dataset

#--------------------------------
# Initialize: Standard Dataset
#--------------------------------

class Dataset:

    def __init__(self, samples, labels, vis = None):

        self.vis = vis
        self.labels = labels
        self.samples = samples

#--------------------------------
# Create: OSR Datasets, ORCA 
#--------------------------------

def combine_data(knowns, unknowns):

    index = np.max(knowns.labels) + 1
    uni_labels = list((np.ones(len(unknowns.labels)) * index).astype(int))

    osr_labels = knowns.labels + uni_labels
    vis_labels = knowns.labels + list((index + np.asarray(unknowns.labels)))

    samples = knowns.samples + unknowns.samples

    return Dataset(samples, osr_labels, vis_labels)

#--------------------------------
# Reduce: Dataset (Train, Valid)
#--------------------------------

def constrain_data(dataset, value, title, rank = 0):

    new_samples, new_labels = [], []

    for current_label in np.unique(dataset.labels):

        indices = np.where(np.asarray(dataset.labels) == current_label)

        samples = np.asarray(dataset.samples)[indices]

        num_samples = int(samples.shape[0] * (value / 100))

        for i, current_sample in enumerate(samples):

            if(i >= num_samples):
                break

            new_samples.append(current_sample)
            new_labels.append(current_label) 
 
    if(rank == 0):
        print("Constrained Number %s: %s" % (title, len(new_samples)))
   
    return Dataset(new_samples, new_labels)

#--------------------------------
# Filter: Dataset, Training
#--------------------------------

def filter_train_data(dataset, known_indices):

    counter = 0
    new_samples, new_labels = [], []
    for current_label in np.unique(dataset.labels):

        if(current_label in known_indices):
            indices = np.where(current_label == np.asarray(dataset.labels))
            class_samples = np.asarray(dataset.samples)[indices]

            for i in range(class_samples.shape[0]):
                new_samples.append(class_samples[i])
                new_labels.append(counter)

            counter = counter + 1

    return Dataset(new_samples, new_labels)

#--------------------------------
# Filter: Dataset (Valid, Test)
#--------------------------------

def filter_test_data(dataset, known_indices):

    new_samples, new_labels, vis_labels = [], [], []
    for current_label in np.unique(dataset.labels):

        indices = np.where(current_label == np.asarray(dataset.labels))
        class_samples = np.asarray(dataset.samples)[indices]
        value = 1 if(current_label in known_indices) else 0

        for i in range(class_samples.shape[0]):
            new_samples.append(class_samples[i])
            new_labels.append(value)
            vis_labels.append(current_label)
            
    return Dataset(new_samples, new_labels, vis_labels)

#--------------------------------
# Loading: Supervised Dataset
#--------------------------------

def gather_data(path, title, rank = 0):

    all_folders = os.listdir(path)
    all_folders.sort()

    if(rank == 0):
        print("\n-------------------------\n")

    all_samples, all_labels = [], []

    total_files = 0
    for i, current_folder in enumerate(all_folders):

        path_folder = os.path.join(path, current_folder)

        if("images" in os.listdir(path_folder)):
            path_folder = os.path.join(path_folder, "images")

        total_files += len(os.listdir(os.path.join(path, path_folder)))

    number = int(total_files * 0.20)

    count = 0
    for i, current_folder in enumerate(all_folders):

        path_folder = os.path.join(path, current_folder)

        if("images" in os.listdir(path_folder)):
            path_folder = os.path.join(path_folder, "images")

        all_files = os.listdir(os.path.join(path, path_folder))
        all_files.sort()

        for current_file in all_files:
            all_samples.append(os.path.join(path_folder, current_file))
            all_labels.append(i)
        
            if(rank == 0 and count % number == 0):
                print("%s : %s percent" % (title, round(count / total_files, 2) * 100))

            count = count + 1

    if(rank == 0):
        print("%s : 100 percent" % (title))

        print("\nOriginal Number Samples: %s" % (len(all_samples)))
        print("Original Number Classes: %s" % (len(np.unique(all_labels))))

        if(not("Train" in title)):
            print("\n-------------------------\n")

    return Dataset(all_samples, all_labels)

#--------------------------------
# Load: ORCA Non-Training Dataset
#--------------------------------

def load_testing(path_data, known_indices, rank = 0):

    return filter_test_data(gather_data(path_data, "Valid", rank), known_indices)

#--------------------------------
# Load: ORCA Training Dataset
#--------------------------------

def load_training(path_train, path_uni, known_indices, constrain = [0, 100], rank = 0):

    knowns = filter_train_data(gather_data(path_train, "Train", rank), known_indices)
    unknowns = gather_data(path_uni, "Uni", rank)

    if(constrain[0]):
        #knowns = constrain_data(knowns, constrain[1], "Train", rank)
        unknowns = constrain_data(unknowns, constrain[1], "Uni", rank)

    return combine_data(knowns, unknowns)
    
#--------------------------------
# Initialize: Pytorch Dataloaders
#--------------------------------

def run(params, rank):

    train = load_training(params["path_train"], params["path_uni"], params["known_indices"], params["constrain"], rank)
    valid = load_testing(params["path_valid"], params["known_indices"], rank)

    transforms = load_data_transforms(params["transforms"], params["interpolate"], params["sample_shape"])
    train = Pytorch_Dataset(train, transforms, params["sample_shape"][0], params["space_size"], "train")
    valid = Pytorch_Dataset(valid, transforms, params["sample_shape"][0], params["space_size"], "valid")

    train = DataLoader(train, batch_size = params["batch_size"], shuffle = 1)
    valid = DataLoader(valid, batch_size = params["batch_size"], shuffle = 0)

    return train, valid

