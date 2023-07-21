import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef

from graph_encoder import GraphKnowEncoder
from knowledge_fusion import KnowFusionModel
from indicator_encoder import IndicatorEncoder
from document_encoder import DocEncoder
from cross_attention import CrossAttentionEncoder

import pdb

class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.graph_encoder = GraphKnowEncoder(device, args.input_graph_dim, args.output_graph_dim)
        self.knowledge_encoder = KnowFusionModel(device, args.input_llm_dim, args.input_graph_dim, args.output_know_dim)
        self.indicator_encoder = IndicatorEncoder(device, args.input_ind_dim, args.output_ind_dim)
        self.document_encoder = DocEncoder(device, args.input_bert_dim, args.output_doc_dim)
        self.cross_att_encoder = CrossAttentionEncoder(device, args.input_att_dim, args.hidden_att_dim, args.output_att_dim, args.num_head)

    def forward(self, input_data, graph, llm_embs, idxs, label, mode):
        node_embs = self.graph_encoder(graph) # 1 * node_num * dim
        graph_emb = torch.index_select(node_embs, 1, torch.tensor(idxs).to(self.device)) # 1 * entity_num * dim
        graph_emb = torch.squeeze(torch.permute(graph_emb, (1,0,2)), 1) # entity * dim
        knowledge_embedding = self.knowledge_encoder(llm_embs, graph_emb)
        pdb.set_trace()
        indicator_embedding = self.indicator_encoder(indicator_seq)
        document_embedding = self.document_encoder(text_seq)

        cross_embedding1 = self.cross_att_encoder(indicator_embedding, document_embedding)
        knowledge_embedding = knowledge_embedding.unsqueeze(0).repeat(indicator_embedding.shape[0],1)
        cross_ebmedding2 = self.cross_att_encoder(cross_embedding1, knowledge_embedding)

        pdb.set_trace()