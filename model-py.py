import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        """
        Initialize the TinyLSTM model
        
        Args:
            vocab_size: Size of the vocabulary (number of unique characters)
            embedding_dim: Dimension of character embeddings
            hidden_dim: Dimension of hidden state in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(TinyLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Character embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of character indices
            hidden: Hidden state (optional)
            
        Returns:
            output: Logits for next character prediction
            hidden: New hidden state
        """
        # Create character embeddings
        embeds = self.embedding(x)
        
        # LSTM layer
        if hidden is None:
            lstm_out, hidden = self.lstm(embeds)
        else:
            lstm_out, hidden = self.lstm(embeds, hidden)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Reshape for linear layer
        output = self.fc(lstm_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """
        Initialize hidden state
        
        Args:
            batch_size: Batch size
            device: Device to initialize tensors on
            
        Returns:
            Tuple of hidden state and cell state
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
    
    def generate(self, char_to_idx, idx_to_char, prime_str='', predict_len=100, temperature=0.8, device='cpu'):
        """
        Generate text using the trained model
        
        Args:
            char_to_idx: Dictionary mapping characters to indices
            idx_to_char: Dictionary mapping indices to characters
            prime_str: Seed text to start generation
            predict_len: Number of characters to generate
            temperature: Controls randomness (lower is more deterministic)
            device: Device to run generation on
            
        Returns:
            Generated text string
        """
        self.eval()  # Set model to evaluation mode
        
        hidden = None
        prime_input = torch.tensor([char_to_idx[char] for char in prime_str], dtype=torch.long).to(device)
        
        # Push the prime string through the model to set up hidden state
        with torch.no_grad():
            for i in range(len(prime_str) - 1):
                _, hidden = self(prime_input[i].unsqueeze(0).unsqueeze(0), hidden)
        
        # Start with the last character of the prime string
        inp = prime_input[-1].unsqueeze(0).unsqueeze(0)
        generated_str = prime_str
        
        # Generate one character at a time
        for i in range(predict_len):
            with torch.no_grad():
                output, hidden = self(inp, hidden)
                
                # Apply temperature to output probabilities
                output_dist = output.data.view(-1).div(temperature).exp()
                top_char = torch.multinomial(output_dist, 1)[0]
                
                # Add predicted character to string and use as next input
                predicted_char = idx_to_char[top_char.item()]
                generated_str += predicted_char
                inp = top_char.unsqueeze(0).unsqueeze(0)
        
        return generated_str
