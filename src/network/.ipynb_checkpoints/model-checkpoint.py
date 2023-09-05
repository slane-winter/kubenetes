#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import torch
import segmentation_models_pytorch as smp

from copy import deepcopy as copy

#--------------------------------
# Model: Create ORCA Network
#--------------------------------

class ORCA(torch.nn.Module):

    def __init__(self, params):

        super().__init__()
    
        # Load: Dataset Parameters

        self.data_shape = list(copy(params["sample_shape"]))
        self.space_size = params["space_size"]

        # Load: Model Parameters

        self.learning_rate = params["learning_rate"]
        self.current_epoch = 0

        # Initialize: Architecture

        self.data_shape.insert(0, 1)
        temp = torch.rand(self.data_shape)

        # - Unet has only 1 class since just using as an autoencoder

        model = smp.Unet(encoder_name = "resnet18", encoder_weights = "imagenet", 
                         in_channels = self.data_shape[1], classes = self.data_shape[1])

        self.encoder = model.encoder
        self.decoder = model.decoder
        self.seg_head = model.segmentation_head

        # - Creating latent space between encoder and decoder of unet architecture

        num_features = self.encoder(temp)[-1].view(-1).shape[0]

        self.encode_space = torch.nn.Linear(num_features, self.space_size)
        self.decode_space = torch.nn.Linear(self.space_size, num_features)

        self.constrain = torch.nn.Sigmoid()

        # - Initialize optimizer
        
        self.configure_optimizer()

        # - Transfer Learn (Resnet)

        for n, l in self.encoder.named_parameters():
            if("conv" in n and not("layer4" in n)):
                l.requires_grad = False

    #----------------------------
    # Objective: Reconstruction
    #----------------------------

    def ae_loss(self, preds, labels):

        fn = torch.nn.MSELoss()
        
        return fn(preds, labels)

    #----------------------------
    # Objective: Pull (Similar)
    #----------------------------

    def pull_loss(self, distances):

        return torch.mean(1 / 2 * torch.pow(distances, 2))

    #----------------------------
    # Create: Objective Function
    #----------------------------

    def objective(self, distances, anchors, recons, samples):

        pull = self.pull_loss(distances)
        reconstruct = self.ae_loss(recons, samples)
        anchor_loss = pull + reconstruct

        return {"total": anchor_loss, "pull": pull, "recon": reconstruct}

    #----------------------------
    # Create: Optimizer Function
    #----------------------------

    def configure_optimizer(self):

        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay = 0.0)

    #----------------------------
    # Create: Model Forward Pass
    #----------------------------

    def forward(self, x, anchors = None):

        # Encoder: Feature Reduction

        x = self.encoder(x)

        # - Gather last layer of Resnet encoder
       
        x_last = x[-1]
        x_shape = x_last.size()
        x_last = x_last.view(x_shape[0], -1)

        # Gather: OSR Features

        features = self.encode_space(x_last)
        features = self.constrain(features)

        # Comparison: Euclidean Distances From Anchors

        if(anchors is not None):
            distances = torch.norm(features - anchors, p = 2, dim = 1)
        else:
            distances = None

        # Decoder: Feature Reconstruction

        x_last = self.decode_space(features)
        x_last = x_last.view(x_shape)

        # - Update last layer of Resnet encoder
       
        x[-1] = x_last

        # - Gather observation reconstructions

        x_recons = self.seg_head(self.decoder(*x))

        return distances, x_recons, features

    #----------------------------
    # Create: Train, Valid Cycle
    #----------------------------

    def epoch_cycle(self, data, title):

        device = next(self.parameters()).device

        for i, batch in enumerate(data):

            samples, anchors, memory = batch

            samples = samples.to(device)
            anchors = anchors.to(device)
            memory = memory.to(device)

            # Evaluate: Dataset Samples

            distances, recons, _ = self(samples, anchors)

            # Calculate: Objective Performance

            loss = self.objective(distances, anchors, recons, memory)

            # Update: Network Parameters

            if(title == "train"):
                self.optimizer.zero_grad()
                loss["total"].backward()
                self.optimizer.step()
                #self.current_epoch += 1

            # Track: Objective Performance

            for current_key in loss.keys():
                loss[current_key] = loss[current_key].item()

            if(i == 0):
                total_loss = loss
            else:
                for current_key in total_loss.keys():
                    total_loss[current_key] += loss[current_key]

        # Finalize: Epoch Analysis

        for current_key in total_loss.keys():
            total_loss[current_key] /= (i + 1)

        if(title == "train"):
            self.current_epoch += 1

        return total_loss

