import torch
import torch.nn as nn

class CrossAttentionEncoder(nn.Module):
    def __init__(self, device, input_dim, hidden_dim, output_dim, num_head):
        super().__init__()
        self.device = device
        self.multihead_attention = MultiHeadAttention(input_dim, hidden_dim, num_head, self.device)
        self.ffn = PoswiseFeedForwardNet(input_dim, output_dim, self.device)

    def forward(self, seq1_inputs, seq2_inputs):
        outputs, attention = self.multihead_attention(seq1_inputs, seq2_inputs, seq2_inputs)
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