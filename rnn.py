import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

#učitavati podatke iz neke csv datoteke ili slično i skalirati do 0 i 1 temp_scaled= (temp_raw-0)/(50-0) temp 

def generate_temp_data():
    # 7 dana, svaka 5 minuta = 2016 mjerenja
    t = np.arange(0, 2016)
    
    # 1. Dnevni ciklus (sinusoida koja se ponavlja svakih 288 koraka)
    daily_cycle = np.sin(2 * np.pi * t / 288) * 10 + 20
    
    # 2. Lagani uzlazni trend kroz tjedan
    trend = 0.005 * t
    
    # 3. Nasumični šum (npr. oblaci, vjetar)
    noise = np.random.normal(0, 0.5, len(t))
    
    temp_raw = daily_cycle + trend + noise
    
    # MIN-MAX SKALIRANJE na [0, 1]
    temp_min = temp_raw.min()
    temp_max = temp_raw.max()
    temp_scaled = (temp_raw - temp_min) / (temp_max - temp_min)
    
    return temp_scaled

# Generiramo podatke
data = generate_temp_data()

# --- 1. PRIPREMA PODATAKA ---

def create_sequences(data, seq_length):
    """
    Prevara ravni niz podataka u prozore (sequences).
    data: list ili np.array (skaliran na 0-1)
    seq_length: koliko prethodnih mjerenja gledamo
    """
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    
    # Prebacivanje u torch tenzore oblika (batch, seq_len, input_size)
    X = torch.from_numpy(np.array(xs)).float().reshape(-1, seq_length, 1)
    y = torch.from_numpy(np.array(ys)).float().reshape(-1, 1)
    return X, y

# --- 2. MODEL ARHITEKTURA ---

class RNN_LSTM_Base(nn.Module):
    def training_step(self, batch):
        samples, targets = batch
        outputs = self(samples.float())
        loss = nn.functional.mse_loss(outputs, targets.float())
        return loss

class VanillaRNN(RNN_LSTM_Base):
    def __init__(self, in_size, hid_size, out_size, n_layers=1):
        super(VanillaRNN, self).__init__()        
        self.input_size = in_size
        self.hidden_size = hid_size
        self.n_layers = n_layers        
        
        # Definiranje RNN sloja
        self.rnn = nn.RNN(in_size, hid_size, n_layers, batch_first=True).float()        
        # Definiranje izlaznog linearnog sloja
        self.linear = nn.Linear(hid_size, out_size).float()
        
    def forward(self, x):
        # Inicijalizacija skrivenog stanja (h0) s nulama
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).float()
        
        # x shape: (batch, seq_len, input_size)
        out, hn = self.rnn(x.float(), h0)        
        
        # Uzimamo zadnje skriveno stanje (zadnji korak u nizu)
        # hn[-1] je oblika (batch, hidden_size)
        last_hidden = hn[-1]
        
        # Aktivacija i finalni prediktor
        out = torch.tanh(last_hidden)
        out = self.linear(out)
        return out

# --- 3. TRENING I PREDIKCIJA ---

def fit(epochs, lr, model, train_loader, opt_func=torch.optim.Adam):
    optimizer = opt_func(model.parameters(), lr)
    model.train()
    
    for epoch in range(epochs):
        epoch_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {np.mean(epoch_losses):.6f}")
            
    return f'Trained for {epochs} epochs'

@torch.no_grad()
def predict(model, dataloader):
    model.eval()
    results = []
    for dl in dataloader:
        sample, target = dl
        output = model(sample.float())
        # .item() radi samo ako je batch_size=1
        results.append([output.cpu().numpy().flatten()[0], target.cpu().numpy().flatten()[0]])
    return np.array(results)

# --- 4. PRIMJER KORIŠTENJA (BOILERPLATE) ---

if __name__ == "__main__":
    # Generiranje lažnih podataka (npr. sinusoida kao temperatura)
    t = np.linspace(0, 100, 1000)
    temp_data = (np.sin(t) + 1) / 2  # Skalirano na [0, 1]
    
    # Postavke prozora: gledamo zadnjih 24 mjerenja (2 sata ako je mjerenje svakih 5 min)
    X, y = create_sequences(temp_data, seq_length=24)
    
    # Podjela na trening i test
    train_ds = TensorDataset(X[:800], y[:800])
    test_ds = TensorDataset(X[800:], y[800:])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1)
    
    # Inicijalizacija modela
    # in_size=1 (samo temp), hid_size=32 (memorija), out_size=1 (temp za 5 min)
    my_model = VanillaRNN(in_size=1, hid_size=32, out_size=1, n_layers=1)
    
    # Trening
    fit(epochs=20, lr=0.001, model=my_model, train_loader=train_loader)
    
    # Predviđanje
    predictions = predict(my_model, test_loader)
    print("Prvih 5 predviđanja [Predviđeno, Stvarno]:")
    print(predictions[:5])

def plot_results(predictions):
    # predictions je array oblika [n, 2] -> [predviđeno, stvarno]
    pred = predictions[:, 0]
    true = predictions[:, 1]
    
    plt.figure(figsize=(15, 6))
    
    # Graf 1: Cijeli testni skup
    plt.subplot(1, 2, 1)
    plt.plot(true, label='Stvarna temperatura', color='blue', alpha=0.7)
    plt.plot(pred, label='Predviđeno', color='red', linestyle='--', alpha=0.8)
    plt.title('Usporedba na testnom skupu')
    plt.xlabel('Vremenski koraci (svakih 5 min)')
    plt.ylabel('Skalirana temperatura')
    plt.legend()
    
    # Graf 2: Zoom na zadnjih 100 mjerenja
    plt.subplot(1, 2, 2)
    plt.plot(true[-100:], label='Stvarna', color='blue')
    plt.plot(pred[-100:], label='Predviđeno', color='red', linestyle='--')
    plt.title('Zoom (Zadnjih 100 mjerenja)')
    plt.xlabel('Vremenski koraci')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. Generiraj kompleksnije podatke (ciklus + trend + šum)
    data = generate_temp_data()
    
    # 2. Postavke prozora
    seq_length = 48  # Gledamo zadnja 2 sata (24 * 5min)
    X, y = create_sequences(data, seq_length)
    
    # 3. Podjela na trening i test (80% trening)
    train_size = int(len(X) * 0.8)
    
    train_ds = TensorDataset(X[:train_size], y[:train_size])
    test_ds = TensorDataset(X[train_size:], y[train_size:])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1)
    
    # 4. Inicijalizacija modela
    # hid_size=64 daje malo više "kapaciteta" za učenje kompleksnijih trendova
    my_model = VanillaRNN(in_size=1, hid_size=64, out_size=1, n_layers=1)
    
    # 5. Trening
    print("Započinjem trening...")
    fit(epochs=30, lr=0.001, model=my_model, train_loader=train_loader)
    
    # 6. Predviđanje i Vizualizacija
    print("Predviđanje na testnim podacima...")
    predictions = predict(my_model, test_loader)
    
    # Poziv funkcije za grafički prikaz
    plot_results(predictions)