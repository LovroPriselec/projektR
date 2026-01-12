import numpy as np
import torch

def scale_temp(temp_raw, t_min=0, t_max=50):
    return (temp_raw - t_min) / (t_max - t_min)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])

    X = torch.tensor(xs).float().reshape(-1, seq_length, 1)
    y = torch.tensor(ys).float().reshape(-1, 1)
    return X, y