import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import openai


class GraphKnowEncoder(nn.Module):
    def __init__(self, device, input_dim, output_dim):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.output_dim = output_dim
        self.num_bases = 2
        self.num_rel = 2
        self.num_layers = 2
        self.dropout = 0.5
        self.layers = nn.ModuleList()
        self.relu = nn.ReLU()
        self.layers.append(RGCN(self.input_dim, self.hidden_dim, self.num_bases, self.num_rel, self.device))
        self.layers.append(RGCN(self.hidden_dim, self.output_dim, self.num_bases, self.num_rel, self.device))

    def forward(self, graph_data):
        embeddings = []
        for t in range(len(graph_data)):
            x = graph_data[t]['features'].to(self.device)
            y = graph_data[t]['adj_list']
            for i, layer in enumerate(self.layers):
                x = layer(x, y)
                x = F.dropout(self.relu(x), self.dropout, training=self.training)
            embeddings.append(x)
        output = torch.stack(embeddings)
        return output
        
class RGCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_bases, num_rel, device, bias=False):
        super(RGCN, self).__init__()
        self.num_bases = num_bases
        self.device = device
        self.num_rel = num_rel
        # R-GCN weights
        if num_bases > 0:
            self.w_bases = Parameter(torch.FloatTensor(num_bases, input_dim, output_dim)).to(device)
            self.w_rel = Parameter(torch.FloatTensor(num_rel, num_bases)).to(device)
        else:
            self.weight = Parameter(torch.FloatTensor(num_rel, input_dim, output_dim)).to(device)
        # R-GCN bias
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_dim)).to(device)
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.num_bases > 0:
            nn.init.xavier_uniform_(self.w_bases.data)
            nn.init.xavier_uniform_(self.w_rel.data)
        else:
            nn.init.xavier_uniform_(self.weight.data)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input, adj_list): 
        if self.num_bases > 0:
            self.weight = torch.einsum("rb, bio -> rio", (self.w_rel, self.w_bases))
        # shape(r*input_size, output_size)
        weights = self.weight.view(self.weight.shape[0] * self.weight.shape[1], self.weight.shape[2])  
        # Each relations * Weight
        supports = []
        for i in range(self.num_rel):
            adj = adj_list[i].to(self.device)
            if input is not None:
                supports.append(torch.sparse.mm(adj.float(), input.float()))
            else:
                supports.append(adj)

        tmp = torch.cat(supports, dim=1)
        # shape(#node, output_size)
        output = torch.mm(tmp.float(), weights)

        if self.bias is not None:
            output += self.bias.unsqueeze(0)
        return output