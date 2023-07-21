import torch
import torch.nn as nn
import numpy as np

import pdb

class FusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, direction_type, source_fusion_type, ts_fusion_type, all_fusion_type, source_num, device):
        super().__init__()
        self.device = device
        self.direction_type = direction_type
        self.cross_source_fusion_model = CSFModel(input_dim, hidden_dim, output_dim, source_fusion_type, source_num, device)
        self.cross_time_fusion_model = CTFModel(input_dim, hidden_dim, output_dim, ts_fusion_type, device)
        if direction_type == 'bi':
            self.all_fusion_model = AFModel(2 * output_dim, output_dim, all_fusion_type, device)
        else:
            self.all_fusion_model = AFModel(output_dim, output_dim, all_fusion_type, device)

    def forward(self, input_embedding):
        st_embedding = None
        ts_embedding = None
        if self.direction_type == 'st' or self.direction_type == 'bi':
            time_embeddings = []
            for time_idx in range(input_embedding.shape[1]):
                csf_embedding = self.cross_source_fusion_model(input_embedding[:,time_idx,:,:])
                time_embeddings.append(csf_embedding)
            st_embedding = self.cross_time_fusion_model(torch.stack(time_embeddings).permute(1,0,2))
        if self.direction_type == 'ts' or self.direction_type == 'bi':
            source_embeddings = []
            for source_idx in range(input_embedding.shape[2]):
                ctf_embedding = self.cross_time_fusion_model(input_embedding[:,:,source_idx,:])
                source_embeddings.append(ctf_embedding)
            ts_embedding = self.cross_source_fusion_model(torch.stack(source_embeddings).permute(1,0,2))

        if self.direction_type == 'bi':
            fused_embedding = self.all_fusion_model(torch.concat([st_embedding, ts_embedding], 1))
        elif self.direction_type == 'ts':
            fused_embedding = self.all_fusion_model(ts_embedding)
        else:
            fused_embedding = self.all_fusion_model(st_embedding)
        return fused_embedding

class CSFModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, fusion_type, source_num, device):
        super().__init__()
        self.device = device
        self.fusion_type = fusion_type
        # transformer fusion module
        if self.fusion_type == 'trans':
            self.layers = nn.ModuleList()
            num_layer = 1
            num_head = 1
            for i in range(num_layer):
                self.layers.append(TransformerEncoderLayer(input_dim, hidden_dim, output_dim, num_head, self.device))
        elif self.fusion_type == 'cat':
            self.layer = nn.Linear(input_dim * source_num, output_dim).to(device)
        elif self.fusion_type == 'expert':
            self.experts = []
            for i in range(source_num):
                self.experts.append(MLPModel(input_dim * source_num, output_dim, device))
            self.gate_layer = nn.Sequential(
                nn.Linear(input_dim * source_num, source_num, bias=False),
                nn.Softmax(dim=1)
                ).to(self.device)

    def forward(self, inputs):
        inputs.to(self.device) # batch_size, seq_len, dim_size
        outputs = inputs
        if self.fusion_type == 'trans':
            attentions = []
            for layer in self.layers:
                outputs, attention = layer(outputs)
                attentions.append(attention)
            outputs = torch.mean(outputs, 1)
        elif self.fusion_type == 'cat':
            tmp = []
            for i in range(inputs.shape[1]):
                tmp.append(inputs[:,i,:])
            outputs = self.layer(torch.cat(tmp, 1))
        elif self.fusion_type == 'expert':
            tmp = []
            for i in range(inputs.shape[1]):
                tmp.append(inputs[:,i,:])
            hidden = torch.cat(tmp, 1)
            expert_outputs = []
            selectors = self.gate_layer(hidden)
            for i in range(len(self.experts)):
                expert_outputs.append(self.experts[i](hidden) * torch.unsqueeze(selectors[:, i], 1))
            outputs = torch.sum(torch.stack(expert_outputs), 0)
        return outputs


class CTFModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, fusion_type, device):
        super().__init__()
        self.device = device
        self.fusion_type = fusion_type
        # transformer fusion module
        if self.fusion_type == 'trans':
            self.layers = nn.ModuleList()
            num_layer = 1
            num_head = 1
            for i in range(num_layer):
                self.layers.append(TransformerEncoderLayer(input_dim, hidden_dim, output_dim, num_head, self.device))
        elif self.fusion_type == 'alstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.att_layer = TransformerEncoderLayer(input_dim, hidden_dim, output_dim, 1, self.device)

    def forward(self, inputs):
        inputs.to(self.device)
        outputs = inputs
        if self.fusion_type == 'trans':
            attentions = []
            for layer in self.layers:
                outputs, attention = layer(outputs)
                attentions.append(attention)
            outputs = torch.mean(outputs, 1)

        elif self.fusion_type == 'alstm':
            outputs, _ = self.rnn(inputs)
            outputs, attention = self.att_layer(outputs)
            outputs = torch.mean(outputs, 1)
        return outputs


class AFModel(nn.Module):
    def __init__(self, input_dim, output_dim, fusion_type, device):
        super().__init__()
        self.device = device
        self.fusion_type = fusion_type
        # mlp fusion module
        if self.fusion_type == 'mlp':
            self.layer = MLPModel(input_dim, output_dim, device)
    
    def forward(self, input):
        output = self.layer(input)
        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_head, device):
        super().__init__()
        self.device = device
        self.multihead_attention = MultiHeadAttention(input_dim, hidden_dim, num_head, self.device)
        self.ffn = PoswiseFeedForwardNet(input_dim, output_dim, self.device)

    def forward(self, inputs):
        outputs, attention = self.multihead_attention(inputs, inputs, inputs)
        outputs = self.ffn(outputs)
        return outputs, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_head, device):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.num_head = num_head
        self.hidden_dim = hidden_dim
        self.W_Q = nn.Linear(input_dim, hidden_dim * num_head, bias=False).to(self.device)
        self.W_K = nn.Linear(input_dim, hidden_dim * num_head, bias=False).to(self.device)
        self.W_V = nn.Linear(input_dim, hidden_dim * num_head, bias=False).to(self.device)
        self.fc = nn.Linear(num_head * hidden_dim, input_dim, bias=False).to(self.device)
        self.attention_layer = ScaledDotProductAttention(input_dim)
        self.norm_layer = nn.LayerNorm(input_dim).to(self.device)

    def forward(self, input_Q, input_K, input_V):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.num_head, self.hidden_dim).transpose(1,2)
        K = self.W_K(input_K).view(batch_size, -1, self.num_head, self.hidden_dim).transpose(1,2)
        V = self.W_V(input_V).view(batch_size, -1, self.num_head, self.hidden_dim).transpose(1,2)

        context, attention = self.attention_layer(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.num_head * self.hidden_dim)
        output = self.fc(context)
        return self.norm_layer(output + residual), attention


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super().__init__()
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
            nn.ReLU(),
            nn.Linear(output_dim, input_dim, bias=False)
        ).to(self.device)
        self.norm_layer = nn.LayerNorm(input_dim).to(self.device)
    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.norm_layer(output + residual)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.hidden_dim)     
        attention = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attention, V)
        return context, attention


class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super().__init__()
        self.device = device
        # mlp fusion module
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=False),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim, bias=False)
        ).to(self.device)
    
    def forward(self, input):
        output = self.layer(input)
        return output