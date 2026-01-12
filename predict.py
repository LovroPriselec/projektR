import torch
from model import VanillaRNN

SEQ_LEN = 48

model = VanillaRNN(1, 64, 1)
model.load_state_dict(torch.load("models/temp_rnn_model.pth"))
model.eval()

def predict_next(buffer):
    x = torch.tensor(buffer).float().reshape(1, SEQ_LEN, 1)
    with torch.no_grad():
        return model(x).item()