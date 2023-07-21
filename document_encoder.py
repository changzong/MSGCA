import torch
import torch.nn as nn

class DocEncoder(nn.Module):
    def __init__(self, device, input_dim, output_dim):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim).to(device)
    
    def forward(self, doc_seq):
        doc_embedding = self.linear(doc_seq.to(self.device)) # timestamp * output_dim
        return doc_embedding
