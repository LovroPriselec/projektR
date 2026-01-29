#predict.py
import torch
import numpy as np
import os
from model import VanillaRNN
import csv

SEQ_LEN = 24
HORIZON = 12
T_MIN = 1
T_MAX = 25

model = VanillaRNN(in_size=1, hid_size=64, out_size=HORIZON)

def load_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "models", "temp_rnn_model.pth")
    
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Model uspješno učitan: {model_path}")
        return True
    except Exception as e:
        print(f"Greška pri učitavanju modela: {e}")
        return False

def predict_future_list(buffer_raw):
    if len(buffer_raw) != SEQ_LEN:
        print(f"Greška: Buffer mora imati točno {SEQ_LEN} elemenata.")
        return None

    scaled_buffer = [(b- T_MIN) / (T_MAX - T_MIN) for b in buffer_raw]
    buffer_np = np.array(scaled_buffer)
    
    x = torch.tensor(buffer_np).float().view(1, SEQ_LEN, 1)
    
    with torch.no_grad():
        out_scaled = model(x).squeeze().numpy()
        
    out_celsius = out_scaled * (T_MAX - T_MIN) + T_MIN
    
    return out_celsius.tolist()
    
def write_to_csv(filename, data_row):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for value in data_row:
            writer.writerow([value])

    
def read_from_csv(filename):
    if not os.path.exists(filename):
        print("Datoteka ne postoji:", filename)
        return []
    with open(filename, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = [int(row[0]) for row in reader if row]
    return rows

def main():
    if load_model():
		
        test_input = read_from_csv("data/readings.csv")
        
        rezultat = predict_future_list(test_input)
        print(test_input)
        
        if rezultat:
            for i, temp in enumerate(rezultat, 1):
                #print(f"Korak {i:2}: {temp:.2f} °C")
                write_to_csv("data/predictions.csv", rezultat)
                
                
if __name__=="__main__":
	main()
