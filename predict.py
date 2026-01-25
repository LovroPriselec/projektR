#predict.py
import torch
import numpy as np
import os
from model import VanillaRNN

# --- KONSTANTE (Mora biti isto kao u train.py) ---
SEQ_LEN = 48
HORIZON = 12
T_MIN = 1
T_MAX = 25

# Inicijalizacija modela (arhitektura mora odgovarati .pth datoteci)
model = VanillaRNN(in_size=1, hid_size=64, out_size=HORIZON)

def load_model():
    """Učitava težine modela na CPU."""
    # Saznajemo putanju foldera u kojem je ova skripta
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "models", "temp_rnn_model.pth")
    
    try:
        # weights_only=False je nužan ako su verzije Pythona na laptopu i Pi-ju različite
        state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✅ Model uspješno učitan: {model_path}")
        return True
    except Exception as e:
        print(f"❌ Greška pri učitavanju modela: {e}")
        return False

def predict_future_list(buffer_raw):
    """
    Ulaz: Lista od 48 sirovih temperatura (Celziji).
    Izlaz: Lista od 12 predviđenih temperatura (Celziji).
    """
    if len(buffer_raw) != SEQ_LEN:
        print(f"Greška: Buffer mora imati točno {SEQ_LEN} elemenata.")
        return None

    # 1. Skaliranje ulaza (Min-Max)
    buffer_np = np.array(buffer_raw)
    scaled_buffer = (buffer_np - T_MIN) / (T_MAX - T_MIN)
    
    # 2. Priprema za model [Batch, Seq, Feature] -> [1, 48, 1]
    x = torch.tensor(scaled_buffer).float().view(1, SEQ_LEN, 1)
    
    # 3. Predviđanje
    with torch.no_grad():
        out_scaled = model(x).squeeze().numpy() # Dobivamo 12 skaliranih vrijednosti
        
    # 4. Odskaliranje (Vraćanje u Celzije)
    out_celsius = out_scaled * (T_MAX - T_MIN) + T_MIN
    
    # Vraćamo kao običnu Python listu radi lakšeg rada u service.py
    return out_celsius.tolist()

# --- TESTIRANJE NA RASPBERRY PI-ju ---
if __name__ == "__main__":
    if load_model():
        # Simuliramo zadnjih 48 očitanja (npr. sve oko 23 stupnja)
        test_input = [23.0 + np.random.uniform(-0.5, 0.5) for _ in range(48)]
        
        rezultat = predict_future_list(test_input)
        
        if rezultat:
            print("\n--- PREDVIĐANJE ZA SLJEDEĆIH 12 KORAKA ---")
            for i, temp in enumerate(rezultat, 1):
                print(f"Korak {i:2}: {temp:.2f} °C")