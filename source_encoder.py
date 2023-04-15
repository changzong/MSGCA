import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import pickle
import jieba

import pdb

class SourceEncoder(nn.Module):
    def __init__(self, input_ind_dim, input_doc_dim, input_graph_dim, output_dim, word2vec_path, device):
        super().__init__()
        self.device = device
        self.doc_encoder = DocEncoder(device, input_doc_dim, output_dim, word2vec_path)
        self.indicator_price_encoder = IndicatorEncoder(device, input_ind_dim, output_dim, 2)
        self.indicator_stats_encoder = IndicatorEncoder(device, input_ind_dim, output_dim, 2)
        self.graph_encoder = GraphEncoder(device, input_graph_dim, output_dim)

    # input_data: [entity, timestamp, source, [value]]
    def forward(self, input_data, graph, idxs):
        embeddings = [] # [entity, timestamp, source, embedding]
        node_embs = self.graph_encoder(graph)
        for entity, idx in zip(input_data, idxs):
            embedding_stamp = []
            for timestamp in entity:
                embedding_source = []
                embedding_source.append(self.doc_encoder(timestamp[0]))
                embedding_source.append(self.indicator_price_encoder(timestamp[1]))
                embedding_source.append(self.indicator_stats_encoder(timestamp[2]))
                embedding_source.append(torch.squeeze(torch.index_select(node_embs, 0, torch.tensor(idx)),0))
                embedding_stamp.append(torch.stack(embedding_source))
            embeddings.append(torch.stack(embedding_stamp))
        embedding_output = torch.stack(embeddings)
        return embedding_output


class DocEncoder(nn.Module):
    def __init__(self, device, input_dim, output_dim, word2vec_path):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        print("Loading word vectors...")
        with open(word2vec_path, 'rb') as f:
            self.wordvec_dict = pickle.load(f)
        self.linear = nn.Linear(input_dim, output_dim).to(device)
        self.doc_adaptor = SourceAdaptor(device, output_dim)

    def doc2vec(self, doc):
        doc_vecs = []
        words = jieba.cut(doc)
        for word in words:
            if word in self.wordvec_dict:
                doc_vecs.append(self.wordvec_dict[word])
        doc_vec = torch.mean(torch.FloatTensor(doc_vecs), 0)
        return doc_vec
    
    def forward(self, doc_data):
        doc_embeddings = []
        if doc_data is None:
            doc_embeddings.append(torch.zeros(self.output_dim))
        else:
            for doc in doc_data:
                doc_embeddings.append(self.linear(self.doc2vec(doc)))
        x = torch.mean(torch.stack(doc_embeddings), 0)
        output = self.doc_adaptor(x)
        return output


class IndicatorEncoder(nn.Module):
    def __init__(self, device, input_dim, output_dim, value_num):
        # value_num: the number of values in indicator input data, such as 2 in [close_price, return_ratio]
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.layers = []
        for i in range(value_num):
            self.layers.append(nn.Linear(input_dim, output_dim).to(device))
        self.indicator_adaptor = SourceAdaptor(device, output_dim)

    def forward(self, indicator_data):
        indicator_embeddings = []
        if indicator_data is None:
            indicator_embeddings.append(torch.zeros(self.output_dim))
        else:
            for i in range(len(indicator_data)):
                ind_tensor = torch.tensor(indicator_data[i]).float().to(self.device)
                indicator_embeddings.append(self.layers[i](torch.unsqueeze(ind_tensor, 0)))
        x = torch.mean(torch.stack(indicator_embeddings),0)
        output = self.indicator_adaptor(x)
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

