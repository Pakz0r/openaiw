import torchvision
import torch.nn as nn
import torch

class HPEnet(nn.Module):
    def __init__(self):
        super(HPEnet, self).__init__()
        print("Loading HPEnet model...")

        self.resnet = torchvision.models.resnet50(weights="ResNet50_Weights.DEFAULT") #ResNet50_Weights.DEFAULT
        self.resnet.fc = nn.Linear(2048, 2048)
        self.fc = nn.Linear(2048, 2048)

        # Classification layers
        self.fc_class = nn.Linear(2048, 1921)
        
        # Regression layers
        self.fc_r1 = nn.Linear(2048, 3)
        self.fc_r2 = nn.Linear(2048, 3)
        self.fc_r3 = nn.Linear(2048, 3)
    
    def forward(self, x):
        # Backbone
        x = self.resnet(x)

        # Dense layer
        x = torch.nn.functional.relu(x)
        x = self.fc(x)

        # Regression layers
        r1 = self.fc_r1(x)
        r2 = self.fc_r2(x)
        r3 = self.fc_r3(x)

        # Classification layers        
        x = torch.nn.functional.relu(x)
        x = self.fc_class(x)

        return r1, r2, r3, x