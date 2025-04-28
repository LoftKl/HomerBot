import torch
import argparse
from model import TinyLSTM

def load_model(model_path, device='cpu'):
    """
    Load trained model from file
    
    Args:
        model_path: Path to saved model file
        device: Device to load model on
    
    Returns:
        model: Loaded TinyLSTM model
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model parameters
    vocab_size = len(checkpoint['vocab'])
    embedding_dim = checkpoint['embedding_dim']
    hidden_dim = checkpoint['hidden_dim']
    num_layers = checkpoint['num_layers']
    dropout = checkpoint['dropout']
    
    # Create model
    model = TinyLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Get character mappings
    char_to_idx = checkpoint['char_to_idx']
    idx_to_char = checkpoint['idx_to_char']
    
    return model, char_to_idx, idx_to_char

def generate_text(model, char_to_idx, idx_to_char, prime_str, predict_len=200, temperature=0.8, device='cpu'):
    """
    Generate text using the trained model
    
    Args:
        model: TinyLSTM model
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        prime_str: Seed text to start generation
        predict_len: Number of characters to generate
        temperature: Controls randomness (lower is more deterministic)
        device: Device to run generation on
    
    Returns:
        Generated text string
    """
    return model.generate(
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        prime_str=prime_str,
        predict_len=predict_len,
        temperature=temperature,
        device=device
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using trained TinyLLM model")
    parser.add_argument('--model', type=str, default='tinyllm_model.pth', help='Path to trained model file')
    parser.add_argument('--prime', type=str, default='The ', help='Prime string to start generation')
    parser.add_argument('--length', type=int, default=200, help='Length of text to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for text generation')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (cpu or cuda)')
    args = parser.parse_args()
    
    # Check for GPU
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, char_to_idx, idx_to_char = load_model(args.model, device)
    
    # Generate text
    generated_text = generate_text(
        model=model,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        prime_str=args.prime,
        predict_len=args.length,
        temperature=args.temperature,
        device=device
    )
    
    print(f"\nGenerated text:\n{generated_text}")
