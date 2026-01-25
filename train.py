#train.py
import torch
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
from model import VanillaRNN
from dataset import create_sequences, scale_temp

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

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {sum(losses)/len(losses):.6f}")

if __name__ == "__main__":
    df = pd.read_csv('podaci.csv') 
    raw_values = df['temperatura'].values 

    #postaviti tmin i tmax
    scaled_data = scale_temp(raw_values, t_min=1, t_max=25)

    SEQ_LEN = 48
    HORIZON=12
    X, y = create_sequences(scaled_data, seq_length=SEQ_LEN, horizon=HORIZON)

    loader = DataLoader(
        TensorDataset(X, y), batch_size=16, shuffle=True
    )

    model = VanillaRNN(1, 64, HORIZON)
    
    print(f"Započinjem trening na {len(X)} uzoraka...")
    fit(epochs=200, lr=0.001, model=model, loader=loader)

    if not os.path.exists("models"):
        os.makedirs("models")
        
    torch.save(model.state_dict(), "models/temp_rnn_model.pth")
    print("Datoteka models/temp_rnn_model.pth je uspješno kreirana!")