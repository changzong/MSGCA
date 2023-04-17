import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import pdb

class SourceEncoder(nn.Module):
    def __init__(self, input_ind_dim, input_doc_dim, input_graph_dim, output_dim, word2vec_path, device):
        super().__init__()
        self.device = device
        self.doc_encoder = DocEncoder(device, input_doc_dim, output_dim, word2vec_path)
        self.indicator_price_encoder = IndicatorEncoder(device, input_ind_dim, output_dim)
        self.indicator_stats_encoder = IndicatorEncoder(device, input_ind_dim, output_dim)
        self.graph_encoder = GraphEncoder(device, input_graph_dim, output_dim)
    
    # input_data: [entity, source, timestamp, dim]
    def forward(self, input_data, graph, idxs):
        embeddings = []
        node_embs = self.graph_encoder(graph)
        for entity in input_data:
            tmp = []
            tmp.append(self.doc_encoder(entity[0])) # doc input: (timestamp * dim)
            tmp.append(self.indicator_price_encoder(entity[1])) # price input: (timestamp * dim)
            tmp.append(self.indicator_price_encoder(entity[2])) # stats input: (timestamp * dim)
            embeddings.append(torch.stack(tmp))
        other_emb = torch.stack(embeddings)
        graph_emb = torch.index_select(node_embs, 0, torch.tensor(idxs).to(self.device)) # entity * 1 * dim
        graph_emb = torch.unsqueeze(graph_emb,1).repeat(1, other_emb.shape[2], 1)
        embedding_output = torch.cat((other_emb, torch.unsqueeze(graph_emb, 1)),1) # entity * source * timestamp * dim
        return torch.permute(embedding_output, (0,2,1,3)) # entity * timestamp * source * dim
    

class DocEncoder(nn.Module):
    def __init__(self, device, input_dim, output_dim, word2vec_path):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.linear = nn.Linear(input_dim, output_dim).to(device)
        self.doc_adaptor = SourceAdaptor(device, output_dim)
    
    def forward(self, doc_vec):
        doc_embedding = self.linear(doc_vec.to(self.device))
        output = self.doc_adaptor(doc_embedding)
        return output


class IndicatorEncoder(nn.Module):
    def __init__(self, device, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.linear = nn.Linear(input_dim, output_dim).to(device)
        self.indicator_adaptor = SourceAdaptor(device, output_dim)

    def forward(self, indicator_vec):
        indicator_embedding = self.linear(indicator_vec.to(self.device))
        output = self.indicator_adaptor(indicator_embedding)
        return output


class GraphEncoder(nn.Module):
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
        self.graph_adaptor = SourceAdaptor(device, output_dim)

    def forward(self, graph_data):
        x = graph_data['features'].to(self.device)
        y = graph_data['adj_list']
        for i, layer in enumerate(self.layers):
            x = layer(x, y)
            x = F.dropout(self.relu(x), self.dropout, training=self.training)
        output = self.graph_adaptor(x)
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


class SourceAdaptor(nn.Module):
    def __init__(self, device, input_dim):
        super().__init__()
        self.device = device
        self.down_ffn = nn.Linear(input_dim, int(input_dim/2)).to(device)
        self.relu = nn.ReLU()
        self.up_ffn = nn.Linear(int(input_dim/2), input_dim).to(device)

    def forward(self, input):
        hidden = self.down_ffn(input)
        hidden = self.relu(hidden)
        hidden = self.up_ffn(hidden)
        output = input + hidden
        return output

