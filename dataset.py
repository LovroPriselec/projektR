#dataset.py
import numpy as np
import torch

def scale_temp(temp_raw, t_min=1, t_max=25):
    return (temp_raw - t_min) / (t_max - t_min)

def create_sequences(data, seq_length, horizon):
    xs, ys = [], []
    for i in range(len(data) - seq_length-horizon+1):
        xs.append(data[i:(i+seq_length)])
        ys.append(data[(i+seq_length):(i+seq_length+horizon)])

    xs_np = np.array(xs)
    ys_np=np.array(ys)

    X = torch.from_numpy(xs_np).float().unsqueeze(-1)
    y = torch.from_numpy(ys_np).float()
    return X, y