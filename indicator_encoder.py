import torch
import torch.nn as nn

class IndicatorEncoder(nn.Module):
    def __init__(self, device, input_dim, output_dim):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim).to(device)

    def forward(self, ind_seq):
        ind_embedding = self.linear(ind_seq.to(self.device)) # timestamp * output_dim
        return ind_embedding