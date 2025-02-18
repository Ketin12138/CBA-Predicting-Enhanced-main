import torch
import torch.nn as nn

class ContrastiveModel(nn.Module):
    def __init__(self, input_dim):
        super(ContrastiveModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.fc2 = nn.Linear(100, 25)  
        self.fc3 = nn.Linear(25, 1)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        embed = torch.relu(self.fc2(x))  
        output = self.fc3(embed)  
        return output, embed