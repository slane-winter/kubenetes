# Import: Standard Python Libraries

import os
import torch

# Generate: Pre-trained Algorithm Details

def generate(path_save):

    os.environ["TORCH_HOME"] = path_save

    resnet = torch.hub.load("pytorch/vision:v0.10.0", "resnet152", pretrained = True)
    
    effinet = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_efficientnet_b0", pretrained = True)

    alexnet = torch.hub.load("pytorch/vision:v0.10.0", "alexnet", pretrained = True)

# Main: Create Parameters

if __name__ == "__main__":

    path_save = "/develop/data/pre_trained_models"

    generate(path_save)

