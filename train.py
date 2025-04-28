import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from model import TinyLSTM
import argparse
import time

def load_data(filename):
    """
    Load text data from file
    
    Args:
        filename: Path to text file
    
    Returns:
        text_data: String containing all text in the file
    """
    with open(filename, 'r', encoding='utf-8') as f:
        text_data = f.read()
    return text_data

def create_vocabulary(text):
    """
    Create character-level vocabulary from text
    
    Args:
        text: Input text
    
    Returns:
        vocab: Set of unique characters
        char_to_idx: Dictionary mapping characters to indices
        idx_to_char: Dictionary mapping indices to characters
    """
    # Get unique characters
    vocab = sorted(set(text))
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create mappings
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    idx_to_char = {i: ch for i, ch in enumerate(vocab)}
    
    return vocab, char_to_idx, idx_to_char

def prepare_batches(text, char_to_idx, seq_length=100, batch_size=32):
    """
    Prepare batches of input-output pairs for training
    
    Args:
        text: Input text
        char_to_idx: Character to index mapping
        seq_length: Length of each sequence
        batch_size: Number of sequences in each batch
    
    Returns:
        x_batches: Input batches
        y_batches: Target batches
    """
    # Convert text to indices
    text_encoded = [char_to_idx[ch] for ch in text]
    
    # Calculate total number of batches
    num_batches = (len(text_encoded) - 1) // (batch_size * seq_length)
    
    # Truncate text to fit evenly into batches
    text_encoded = text_encoded[:num_batches * batch_size * seq_length + 1]
    
    # Reshape data
    x_data = text_encoded[:-1]  # Input data (all characters except the last one)
    y_data = text_encoded[1:]   # Target data (all characters except the first one)
    
    # Prepare batches
    x_batches = []
    y_batches = []
    
    for i in range(0, len(x_data) - seq_length, seq_length):
        if i + seq_length + batch_size * seq_length <= len(x_data):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                offset = b * seq_length
                x_batch.append(x_data[i + offset:i + offset + seq_length])
                y_batch.append(y_data[i + offset:i + offset + seq_length])
            x_batches.append(torch.tensor(x_batch, dtype=torch.long))
            y_batches.append(torch.tensor(y_batch, dtype=torch.long))
    
    return x_batches, y_batches

def train_model(model, x_batches, y_batches, epochs=30, lr=0.001, clip_value=5.0, device='cpu'):
    """
    Train the model
    
    Args:
        model: TinyLSTM model
        x_batches: Input batches
        y_batches: Target batches
        epochs: Number of training epochs
        lr: Learning rate
        clip_value: Gradient clipping threshold
        device: Device to train on
    
    Returns:
        losses: List of average losses per epoch
    """
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        hidden = None
        start_time = time.time()
        
        for i, (x, y) in enumerate(zip(x_batches, y_batches)):
            x = x.to(device)
            y = y.to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Reset hidden state at the beginning of each batch
            hidden = model.init_hidden(x.size(0), device)
            
            # Forward pass
            output, hidden = model(x, hidden)
            
            # Detach hidden state from history
            hidden = tuple([h.detach() for h in hidden])
            
            # Calculate loss
            batch_loss = criterion(output.transpose(1, 2), y)
            
            # Backward pass
            batch_loss.backward()
            
            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            # Update weights
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(x_batches)}, Loss: {batch_loss.item():.4f}")
        
        # Calculate average loss
        avg_loss = epoch_loss / len(x_batches)
        losses.append(avg_loss)
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f"Epoch {epoch+1}/{epochs} completed, Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # Generate sample text every few epochs
        if (epoch + 1) % 5 == 0:
            sample_text = model.generate(char_to_idx, idx_to_char, prime_str="The ", predict_len=100, device=device)
            print(f"Sample text:\n{sample_text}\n")
    
    return losses

def plot_loss(losses, save_path=None):
    """
    Plot training loss
    
    Args:
        losses: List of average losses per epoch
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Loss plot saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tiny LSTM language model")
    parser.add_argument('--data', type=str, default='data.txt', help='Path to training data file')
    parser.add_argument('--seq_length', type=int, default=100, help='Sequence length for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to train on (cpu or cuda)')
    args = parser.parse_args()
    
    # Check for GPU
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    text_data = load_data(args.data)
    print(f"Text length: {len(text_data)} characters")
    
    # Create vocabulary
    vocab, char_to_idx, idx_to_char = create_vocabulary(text_data)
    
    # Prepare batches
    x_batches, y_batches = prepare_batches(text_data, char_to_idx, args.seq_length, args.batch_size)
    print(f"Number of batches: {len(x_batches)}")
    
    # Create model
    model = TinyLSTM(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    print(model)
    
    # Train model
    losses = train_model(
        model=model,
        x_batches=x_batches,
        y_batches=y_batches,
        epochs=args.epochs,
        lr=args.lr,
        device=device
    )
    
    # Plot loss
    plot_loss(losses, save_path='loss.png')
    
    # Save model
    model_path = 'tinyllm_model.pth'
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout
    }
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}")
