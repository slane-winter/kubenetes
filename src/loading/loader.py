#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import torch
import numpy as np                                                                          
import torch.utils.data as tech 

from PIL import Image

#--------------------------------
# Format: Data, Pytorch Dataset
#--------------------------------

class Dataset(tech.Dataset):

    def __init__(self, data, transforms, num_channels, mode):

        self.mode = mode
        self.data = data
        self.transforms = transforms        
        self.num_channels = num_channels

        # Create: Datasets 

        self.make_dataset()                

    #----------------------------
    # Populate: Testing Dataset
    #----------------------------

    def make_dataset(self):

        self.dataset = [[sample, label] for sample, label in zip(self.data.samples, self.data.labels)]

    #----------------------------
    # Load: Samples, Image Files 
    #----------------------------

    def image_loader(self, data, make_gray = 0):

        image = Image.open(data) 

        if(make_gray):
            tag = "L"
        else: 
            tag = "L" if(self.num_channels == 1) else "RGB"

        image = image.convert(tag)

        return np.asarray(image)

    #----------------------------
    # Gather: Nth Sample, Dataset
    #----------------------------

    def __getitem__(self, index):
      
        # Gather: Samples
 
        samples, labels = self.dataset[index]

        samples = self.image_loader(samples)
        samples = self.transforms[self.mode](image = samples)["image"].float()

        return samples, labels
    
    #---------------------------------
    # Gather: Number Dataset Samples
    #---------------------------------

    def __len__(self):
 
        return len(self.dataset) 

