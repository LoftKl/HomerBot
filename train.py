import torch
import torch.nn as nn
import torch.optim as optim
from model import TinyLSTM
import pickle

# Load dataset
with open('homer-simpson-quotes-dataset.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)

vocab_size = len(chars)
seq_length = 20
batch_size = 200
embed_dim = 128
hidden_dim = 128
num_epochs = 55
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_batch():
    ix = torch.randint(0, len(encoded) - seq_length - 1, (batch_size,))
    x = torch.stack([encoded[i:i+seq_length] for i in ix])
    y = torch.stack([encoded[i+1:i+seq_length+1] for i in ix])
    return x.to(device), y.to(device)

model = TinyLSTM(vocab_size, embed_dim, hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for _ in range(100):  # 100 batches per epoch
        x_batch, y_batch = get_batch()
        optimizer.zero_grad()
        logits, _ = model(x_batch)
        loss = loss_fn(logits.view(-1, vocab_size), y_batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/100:.4f}")

# Save model and tokenizer
torch.save(model.state_dict(), 'tinyllm.pth')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump((stoi, itos), f)
