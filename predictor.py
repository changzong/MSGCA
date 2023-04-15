import torch
import torch.nn as nn

class ScorePredictor(nn.Module):
    def __init__(self, input_dim, output_dim, use_spike_predictor, device):
        super().__init__()
        self.device = device
        if use_spike_predictor:
            self.spike_layer = SpikeNNModel(input_dim, input_dim, input_dim)
        else:
            self.spike_layer = None
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, int(input_dim/2)),
            nn.ReLU(),
            nn.Linear(int(input_dim/2), output_dim, bias=False),
            nn.Sigmoid()
        ).to(self.device)

    def forward(self, fusion_input):
        fusion_input.to(self.device)
        if self.spike_layer:
            fusion_input = self.spike_layer(fusion_input)
        score = self.scorer(fusion_input)
        return score


class SpikeNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpikeNNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.input_hidden_weights = nn.Parameter(torch.randn(input_dim, hidden_dim))
        self.hidden_output_weights = nn.Parameter(torch.randn(hidden_dim, output_dim))

        self.threshold = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x):
        x = torch.where(x > self.threshold, torch.ones_like(x), torch.zeros_like(x))

        hidden_output = torch.matmul(x, self.input_hidden_weights)
        hidden_output = torch.where(hidden_output > self.threshold, torch.ones_like(hidden_output), torch.zeros_like(hidden_output))

        output = torch.matmul(hidden_output, self.hidden_output_weights)

        return output