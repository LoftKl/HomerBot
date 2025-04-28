import torch
from model import TinyLSTM
import pickle

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    stoi, itos = pickle.load(f)

vocab_size = len(stoi)

# Load model
device = torch.device('cpu')
model = TinyLSTM(vocab_size)
model.load_state_dict(torch.load('tinyllm.pth', map_location=device))
model.eval()

def generate(prompt, length=200):
    input_ids = torch.tensor([stoi.get(c, 0) for c in prompt], dtype=torch.long, device=device).unsqueeze(0)
    hidden = None
    output_text = prompt

    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(input_ids, hidden)
            last_logits = logits[:, -1, :]  # (batch, vocab)
            probs = torch.softmax(last_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # shape (batch_size, 1)
            output_text += itos[next_id.item()]
            input_ids = next_id  # <-- FIX: no unsqueeze

    return output_text
