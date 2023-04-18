import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef

from source_encoder import SourceEncoder
from fusion_model import FusionModel
from predictor import ScorePredictor

import pdb

class Model(nn.Module):
    def __init__(self, args, sources, device):
        super().__init__()
        self.args = args
        self.device = device
        self.encoder = SourceEncoder(args.input_ind_dim, args.input_doc_dim, args.input_graph_dim, args.hidden_dim, args.data_path + args.word2vec, self.device)
        self.fusion_model = FusionModel(args.hidden_dim, args.hidden_dim, args.fusion_dim, args.direction_type, args.source_fusion_type, args.ts_fusion_type, args.all_fusion_type, sources, self.device)
        self.predictor = ScorePredictor(args.fusion_dim, args.score_dim, args.use_spike_predictor, self.device)

    def loss_function(self, output, target):
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, torch.tensor(target).to(self.device))
        return loss

    def evaluate(self, output, target):
        output_labels = torch.argmax(output, dim=1)
        acc = accuracy_score(output_labels.cpu().numpy(), np.array(target))
        mcc = matthews_corrcoef(output_labels.cpu().numpy(), np.array(target))
        return acc, mcc

    def forward(self, input_data, graph, idxs, label, mode):
        # shap: sample_num * timestamp_num * source_num * embedding_dim
        source_embedding = self.encoder(input_data, graph, idxs)
        fusion_embedding = self.fusion_model(source_embedding)
        predict_score = self.predictor(fusion_embedding)
        future_days = 1
        target = []
        for item in label:
            target.append(item[future_days-1][0]) # 0,1,2 class indices

        if mode == 'train':
            loss = self.loss_function(predict_score, target)
            return loss
        elif mode == 'test':
            acc, mcc  = self.evaluate(predict_score, target)
            return acc, mcc
