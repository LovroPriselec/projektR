import torch
import torch.nn as nn

class RNN_LSTM_Base(nn.Module):
    def training_step(self, batch):
        samples, targets = batch
        outputs = self(samples.float())
        loss = nn.functional.mse_loss(outputs, targets.float())
        return loss

class VanillaRNN(RNN_LSTM_Base):
    def __init__(self, in_size, hid_size, out_size, n_layers=1):
        super().__init__()
        self.hidden_size = hid_size
        self.n_layers = n_layers

        self.rnn = nn.RNN(
            in_size, hid_size, n_layers, batch_first=True
        )
        self.linear = nn.Linear(hid_size, out_size)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size)
        _, hn = self.rnn(x, h0)
        out = torch.tanh(hn[-1])
        return self.linear(out)