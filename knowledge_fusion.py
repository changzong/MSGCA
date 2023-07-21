import torch
import torch.nn as nn
import pdb

class KnowFusionModel(nn.Module):
    def __init__(self, device, input_dim1, input_dim2, output_dim):
        super().__init__()
        self.device = device
        self.linear1 = nn.Linear(input_dim1, output_dim).to(device)
        self.linear2 = nn.Linear(input_dim2, output_dim).to(device)
        self.layer = nn.GLU()
    
    def forward(self, emb1, emb2):
        hidden1 = self.linear1(emb1.to(self.device))
        hidden2 = self.linear2(emb2.to(self.device))
        hidden = torch.permute(torch.stack([hidden1, hidden2]), (1, 2, 0)) # batch * output_dim * 2
        output = self.layer(hidden) # batch * output_dim * 1
        return torch.squeeze(output, 2) # batch * output_dim
