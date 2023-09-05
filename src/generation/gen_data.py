# Import: Standard Python Libraries

import os
import shutil
import numpy as np
import torchvision

from PIL import Image
from tqdm import tqdm

# Create: Organized Dataset

class Dataset:

    def __init__(self, samples, labels):

        self.samples = samples
        self.labels = labels

# Create: Specified Folder

def create_folder(path):

    if(os.path.exists(path)):
        shutil.rmtree(path)

    os.makedirs(path)

# Save: Single Partition

def save_partition(data, path, class_size = 4, sample_size = 6):

    print("\nSaving Dataset: Path = %s\n" % path)

    for current_class in tqdm(np.unique(data.labels), "Processing"):

        indices = np.where(data.labels == current_class)

        class_samples = data.samples[indices]

        path_folder = os.path.join(path, str(current_class).zfill(class_size))

        create_folder(path_folder)

        for i, sample in enumerate(class_samples):

            path_file = os.path.join(path_folder, str(i).zfill(sample_size) + ".png")

            sample = Image.fromarray(sample)

            sample.save(path_file)

# Save: Image Dataset Partitions

def save_data(train, valid, test, path):

    save_partition(train, os.path.join(path, "train"))
    save_partition(valid, os.path.join(path, "valid"))
    save_partition(test, os.path.join(path, "test"))

    print("\nSaving - Complete\n")

def get_validation(train, percent = 20):

    num_samples = int(len(train.samples) * (percent / 100))

    valid_samples, valid_labels = train.samples[:num_samples], train.labels[:num_samples]
    valid = Dataset(valid_samples, valid_labels)

    train.samples, train.labels = train.samples[num_samples:], train.labels[num_samples:]
    
    return train, valid

# Download: STL 10 Partition

def download_stl(path, split):

    data = torchvision.datasets.STL10(root = path, split = split, download = True)

    samples = np.swapaxes(np.swapaxes(data.data, 1, 3), 1, 2)

    return Dataset(samples, data.labels)

# Download: Cifar 10 Partition

def download_cifar(path, train):

    data = torchvision.datasets.CIFAR10(root = path, train = train, download = True)

    return Dataset(data.data, np.asarray(data.targets))

# Load: STL 10 Dataset

def load_stl(path):

    path = os.path.join(path, "raw_files")

    train = download_stl(path, "train")

    train, valid = get_validation(train)

    test = download_stl(path, "test")

    return train, valid, test

# Load: Cifar 10 Dataset

def load_cifar(path):

    path = os.path.join(path, "raw_files")

    train = download_cifar(path, train = True)

    train, valid = get_validation(train)

    test = download_cifar(path, train = False)

    return train, valid, test

# Generate: Demo Datasets

def generate(path_save):

    # Gather: Cifar 10 

    path = os.path.join(path_save, "cifar")

    train, valid, test = load_cifar(path)

    save_data(train, valid, test, path)
    
    # Gather: STL 10

    path = os.path.join(path_save, "stl0")

    train, valid, test = load_stl(path)

    save_data(train, valid, test, path)

# Main: Create Parameters

if __name__ == "__main__":

    path_save = "/develop/data"
    
    generate(path_save)
    
