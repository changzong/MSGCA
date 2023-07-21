import torch
import torch.nn as nn

class Predictor(nn.Module):
    # input: batch * seq * dim, output: batch * 3
    def __init__(self, device, input_dim, output_dim):
        super().__init__()
        self.device = device
        self.rnn = nn.GRU(input_dim, input_dim, 2, batch_first=True).to(self.device)
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, int(input_dim/2)),
            nn.ReLU(),
            nn.Linear(int(input_dim/2), output_dim, bias=False),
            nn.Softmax(dim=1)
        ).to(self.device)

    def forward(self, fusion_input):
        fusion_input.to(self.device)
        _, last = self.rnn(fusion_input)
        output = self.scorer(last[-1])
        return output