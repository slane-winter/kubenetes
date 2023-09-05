#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import torch

#--------------------------------
# Model: Pre-Trained Classifier
#--------------------------------

class Classifier(torch.nn.Module):

    def __init__(self, params):

        super().__init__()
    
        # Load: Dataset Parameters

        self.learning_rate = params["learning_rate"]
        self.current_epoch = 0

        # - Select architecture

        self.select_architecture(params["arch"], params["sample_shape"][0], params["num_classes"])

        # - Initialize optimizer
        
        self.configure_optimizer()

    #----------------------------
    # Selection: Net Architecture
    #----------------------------

    def select_architecture(self, choice, in_channels, num_classes):

        # Load: ResNet 152 (60M Parameters)

        if(choice == 0):

            self.architecture = torch.hub.load("pytorch/vision:v0.10.0", "resnet152", pretrained = False)

            in_features = self.architecture.fc.in_features
            self.architecture.fc = torch.nn.Linear(in_features, num_classes)

            all_targets = ["layer4", "fc"]

        # Load: EfficientNet B0 (7.1M Parameters)

        elif(choice == 1):

            self.architecture = torch.hub.load("NVIDIA/DeepLearningExamples:torchhub", "nvidia_efficientnet_b0", pretrained = False)

            all_targets = []

            in_features = self.architecture.classifier.fc.in_features
            self.architecture.classifier.fc = torch.nn.Linear(in_features, num_classes)

            all_targets = ["layers.6", "features", "classifier"]

        # Load: Alexnet (61M Parameters)

        elif(choice == 2):

            self.architecture = torch.hub.load("pytorch/vision:v0.10.0", "alexnet", pretrained = False)

            in_features = self.architecture.classifier[-1].in_features
            self.architecture.classifier[-1] = torch.nn.Linear(in_features, num_classes)

            all_targets = ["classifier"]

        # Transfer Learning: Turn Off Gradients 

        for i, (n, l) in enumerate(self.architecture.named_parameters()):
            l.requires_grad = False
            for target in all_targets:
                if(target in n):
                    l.requires_grad = True

    #----------------------------
    # Initialize: Cross Entropy
    #----------------------------

    def classi_loss(self, preds, labels):

        obj = torch.nn.CrossEntropyLoss(label_smoothing = 0.2)
        
        return obj(preds, labels)

    #----------------------------
    # Create: Objective Function
    #----------------------------

    def objective(self, preds, labels):

        classi = self.classi_loss(preds, labels)

        return {"total": classi}

    #----------------------------
    # Create: Optimizer Function
    #----------------------------

    def configure_optimizer(self):

        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)

    #----------------------------
    # Create: Model Forward Pass
    #----------------------------

    def forward(self, x):

        return self.architecture(x)

    #----------------------------
    # Create: Prediction Cycle
    #----------------------------

    def test_cycle(self, data):

        device = next(self.parameters()).device

        results = {"preds": [], "truths": []}

        for i, batch in enumerate(data):

            samples, labels = batch

            samples = samples.to(device)

            # - Evaluate dataset samples

            preds = self(samples).detach().to("cpu")

            results["preds"].append(preds)
            results["truths"].append(labels)
             
        return results

    #----------------------------
    # Create: Train, Valid Cycle
    #----------------------------

    def epoch_cycle(self, data, title):

        device = next(self.parameters()).device

        for i, batch in enumerate(data):

            samples, labels = batch

            samples = samples.to(device)
            labels = labels.to(device)

            # - Evaluate dataset samples

            if(title == "train"):
                self.optimizer.zero_grad()

            preds = self(samples)

            # - Calculate objective performance

            loss = self.objective(preds, labels)

            # - Update: Network Parameters

            if(title == "train"):
                loss["total"].backward()
                self.optimizer.step()

            # - Track objective performance

            for current_key in loss.keys():
                loss[current_key] = loss[current_key].item()

            if(i == 0):
                total_loss = loss
            else:
                for current_key in total_loss.keys():
                    total_loss[current_key] += loss[current_key]

        # Finalize: Epoch Analysis

        self.current_epoch += 1
        
        for current_key in total_loss.keys():
            total_loss[current_key] /= (i + 1)

        return total_loss
