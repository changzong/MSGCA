import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef

from graph_encoder import GraphKnowEncoder
from knowledge_fusion import KnowFusionModel
from indicator_encoder import IndicatorEncoder
from document_encoder import DocEncoder
from cross_attention import CrossAttentionEncoder
from predictor import Predictor

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
        self.cross_att_encoder1 = CrossAttentionEncoder(device, args.input_att_dim, args.hidden_att_dim, args.output_att_dim, args.num_head)
        self.cross_att_encoder2 = CrossAttentionEncoder(device, args.input_att_dim, args.hidden_att_dim, args.output_att_dim, args.num_head)
        self.predictor = Predictor(device, args.input_pred_dim, args.output_pred_dim)

    def loss_function(self, output, target):
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, torch.tensor(target).to(self.device))
        return loss

    def evaluate(self, output, target):
        output_labels = torch.argmax(output, dim=1)
        acc = accuracy_score(output_labels.cpu().numpy(), np.array(target))
        mcc = matthews_corrcoef(output_labels.cpu().numpy(), np.array(target))
        return acc, mcc

    def forward(self, input_data, graph, llm_embs, idxs, label, mode):
        node_embs = self.graph_encoder(graph) # 1 * node_num * dim
        graph_emb = torch.index_select(node_embs, 1, torch.tensor(idxs).to(self.device)) # 1 * batch * dim
        graph_emb = torch.squeeze(torch.permute(graph_emb, (1,0,2)), 1) # batch * dim
        knowledge_embedding = self.knowledge_encoder(llm_embs, graph_emb) # batch * dim
        
        ind_seq = torch.stack([item[1] for item in input_data])
        doc_seq = torch.stack([item[0] for item in input_data])
        indicator_embedding = self.indicator_encoder(ind_seq) # batch * seq * dim
        document_embedding = self.document_encoder(doc_seq) # batch * seq * dim

        cross_embedding1, _ = self.cross_att_encoder1(indicator_embedding, document_embedding)
        # repeat knowledge embedding along with timestamp
        knowledge_embedding = knowledge_embedding.unsqueeze(1).repeat(1,indicator_embedding.shape[1],1)
        cross_ebmedding2, _ = self.cross_att_encoder2(cross_embedding1, knowledge_embedding)

        output_score = self.predictor(cross_ebmedding2) # batch * 3

        future_days = 1
        target = []
        for item in label:
            target.append(item[future_days-1][0]) # 0,1,2 class indices

        if mode == 'train':
            loss = self.loss_function(output_score, target)
            return loss
        elif mode == 'test':
            acc, mcc  = self.evaluate(output_score, target)
            return acc, mcc