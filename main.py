import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv

# Force GPU if available, fallback to CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ CUDA not available. Using CPU.")

input_file = 'homer_quotes.csv'
quotes = []

with open(input_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        quotes.append(row['spoken_words'])  # Use the correct column name

text = ' '.join(quotes)

# Tokenization
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

# Hyperparameters
vocab_size = len(chars)
block_size = 32
batch_size = 16
embedding_dim = 64
hidden_dim = 128
epochs = 1

# Dataset
class HomerDataset(Dataset):
    def __init__(self, text):
        self.data = encode(text)

    def __len__(self):
        return len(self.data) - block_size

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+block_size])
        y = torch.tensor(self.data[idx+1:idx+block_size+1])
        return x, y

dataset = HomerDataset(text)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model
class HomerLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        return self.fc(out)

model = HomerLSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss()

# Training
for epoch in range(epochs):
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")

# Generate quote
def generate(model, start_text="D'oh", length=100):
    model.eval()
    context = torch.tensor(encode(start_text[-block_size:]), dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(length):
            logits = model(context[:, -block_size:])
            next_id = torch.multinomial(torch.softmax(logits[:, -1, :], dim=-1), num_samples=1)
            context = torch.cat([context, next_id], dim=1)
        return decode(context[0].tolist())

print(generate(model, "Woo-hoo!", 100))
