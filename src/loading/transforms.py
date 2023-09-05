#---------------------------------
# Import: Custom Python Libraries
#---------------------------------

import albumentations as album

from albumentations.pytorch import ToTensorV2

#---------------------------------
# Initialize: Dataset Transforms
#---------------------------------

def load_data_transforms(choice, interpolate, data_shape):

    num_channels, height, width = data_shape

    std = tuple([0.5] * num_channels)
    mean = tuple([0.5] * num_channels)
  
    results = {}
    transforms = []
 
    if(choice == 0): 

        if(interpolate):
            transforms.append(album.Resize(height, width))            
        
        other = [album.Normalize(mean = mean, std = std),
                 ToTensorV2()]

        transforms = transforms + other

        results["train"] = album.Compose(transforms)
        results["valid"] = album.Compose(transforms)

    elif(choice == 1):

        if(interpolate):
            transforms.append(album.SmallestMaxSize(max_size = max(height, width) + 32))

        other_train = [album.ShiftScaleRotate(shift_limit = 0.05, scale_limit = 0.05, rotate_limit = 15, p = 0.5),
                       album.RandomCrop(height = height, width = width),
                       #album.RGBShift(r_shift_limit = 15, g_shift_limit = 15, b_shift_limit = 15, p = 0.5),
                       album.RandomBrightnessContrast(p = 0.5),
                       album.Normalize(mean = mean, std = std),
                       ToTensorV2()]

        other_valid = [album.CenterCrop(height = height, width = width),
                       album.Normalize(mean = mean, std = std),
                       ToTensorV2()]

        train_transforms = transforms + other_train
        valid_transforms = transforms + other_valid

        results["train"] = album.Compose(train_transforms)
        results["valid"] = album.Compose(valid_transforms)

    elif(choice == 2):

        if(interpolate):
            transforms.append(album.SmallestMaxSize(max_size = height + 32))

        other = [album.CenterCrop(height = height, width = width),
                 album.Normalize(mean = mean, std = std),
                 ToTensorV2()]

        transforms = transforms + other

        results["train"] = None
        results["valid"] = album.Compose(transforms)

    return results
