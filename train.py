import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import VanillaRNN
from dataset import create_sequences

def fit(epochs, lr, model, loader):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    model.train()

    for epoch in range(epochs):
        losses = []
        for batch in loader:
            optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}: {sum(losses)/len(losses):.6f}")

if __name__ == "__main__":
    # primjer sa simulacijom
    data = np.sin(np.linspace(0, 50, 2000))
    X, y = create_sequences(data, seq_length=48)

    loader = DataLoader(
        TensorDataset(X, y), batch_size=32, shuffle=True
    )

    model = VanillaRNN(1, 64, 1)
    fit(30, 0.001, model, loader)

    torch.save(model.state_dict(), "models/temp_rnn_model.pth")