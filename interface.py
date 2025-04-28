import os
import torch
import argparse
from flask import Flask, render_template, request, jsonify
from model import TinyLSTM

app = Flask(__name__)

# Global variables for model and character mappings
model = None
char_to_idx = None
idx_to_char = None
device = None

def load_model(model_path, device_name='cpu'):
    """
    Load trained model from file
    
    Args:
        model_path: Path to saved model file
        device_name: Device to load model on
    
    Returns:
        model: Loaded TinyLSTM model
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
    """
    global model, char_to_idx, idx_to_char, device
    
    # Set device
    device = torch.device(device_name if torch.cuda.is_available() and device_name == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
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
    
    print("Model loaded successfully!")

def generate_text(prime_str, predict_len=200, temperature=0.8):
    """
    Generate text using the trained model
    
    Args:
        prime_str: Seed text to start generation
        predict_len: Number of characters to generate
        temperature: Controls randomness (lower is more deterministic)
    
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

# Create templates directory and HTML template
def create_templates():
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TinyLLM Text Generator</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .container {
                background-color: #f9f9f9;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            textarea, input[type="number"], input[type="range"] {
                width: 100%;
                padding: 8px;
                margin-bottom: 15px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }
            textarea {
                height: 100px;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            button:hover {
                background-color: #45a049;
            }
            .result {
                margin-top: 20px;
                padding: 15px;
                background-color: #e9f7ef;
                border-radius: 4px;
                white-space: pre-wrap;
            }
            .slider-container {
                display: flex;
                align-items: center;
            }
            .slider-container input {
                flex: 1;
                margin-right: 10px;
            }
            .slider-value {
                width: 50px;
                text-align: center;
            }
            .loading {
                text-align: center;
                margin-top: 20px;
                display: none;
            }
            .footer {
                margin-top: 30px;
                text-align: center;
                font-size: 0.8em;
                color: #666;
            }
        </style>
    </head>
    <body>
        <h1>TinyLLM Text Generator</h1>
        <div class="container">
            <form id="generate-form">
                <div>
                    <label for="prompt">Enter your prompt:</label>
                    <textarea id="prompt" name="prompt" required>The </textarea>
                </div>
                <div>
                    <label for="length">Length of generated text:</label>
                    <input type="number" id="length" name="length" min="10" max="500" value="200">
                </div>
                <div>
                    <label for="temperature">Temperature (randomness):</label>
                    <div class="slider-container">
                        <input type="range" id="temperature" name="temperature" min="0.1" max="2.0" step="0.1" value="0.8">
                        <span id="temp-value" class="slider-value">0.8</span>
                    </div>
                </div>
                <button type="submit">Generate Text</button>
            </form>
            <div class="loading" id="loading">
                <p>Generating text... Please wait.</p>
            </div>
            <div class="result" id="result" style="display: none;">
                <h3>Generated Text:</h3>
                <div id="generated-text"></div>
            </div>
        </div>
        <div class="footer">
            <p>TinyLLM - A character-level language model created for CSCI 4220 - Introduction to Artificial Intelligence</p>
        </div>

        <script>
            // Update temperature value display
            document.getElementById('temperature').addEventListener('input', function() {
                document.getElementById('temp-value').textContent = this.value;
            });

            // Handle form submission
            document.getElementById('generate-form').addEventListener('submit', function(e) {
                e.preventDefault();
                
                // Show loading indicator
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                
                // Get form data
                const prompt = document.getElementById('prompt').value;
                const length = document.getElementById('length').value;
                const temperature = document.getElementById('temperature').value;
                
                // Send request to server
                fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        prompt: prompt,
                        length: parseInt(length),
                        temperature: parseFloat(temperature)
                    }),
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading indicator
                    document.getElementById('loading').style.display = 'none';
                    
                    // Display result
                    document.getElementById('generated-text').textContent = data.text;
                    document.getElementById('result').style.display = 'block';
                })
                .catch((error) => {
                    // Hide loading indicator
                    document.getElementById('loading').style.display = 'none';
                    
                    // Display error
                    document.getElementById('generated-text').textContent = 'Error: ' + error;
                    document.getElementById('result').style.display = 'block';
                });
            });
        </script>
    </body>
    </html>
    """
    
    with open('templates/index.html', 'w') as f:
        f.write(html_content)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def api_generate():
    data = request.json
    prompt = data.get('prompt', 'The ')
    length = int(data.get('length', 200))
    temperature = float(data.get('temperature', 0.8))
    
    # Cap length to prevent resource issues
    length = min(length, 500)
    
    # Generate text
    generated_text = generate_text(prompt, length, temperature)
    
    return jsonify({'text': generated_text})

def run_cli():
    """
    Run the command line interface
    """
    print("\n===== TinyLLM Text Generator =====")
    print("Enter 'quit' or 'exit' to end the program.")
    
    while True:
        prompt = input("\nEnter your prompt (or 'quit' to exit): ")
        
        if prompt.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        length = input("Enter length of text to generate (default: 200): ")
        length = int(length) if length.isdigit() else 200
        
        temperature = input("Enter temperature (0.1-2.0, default: 0.8): ")
        try:
            temperature = float(temperature)
            if temperature < 0.1 or temperature > 2.0:
                temperature = 0.8
        except ValueError:
            temperature = 0.8
        
        print("\nGenerating text... Please wait.")
        generated_text = generate_text(prompt, length, temperature)
        
        print("\n===== Generated Text =====")
        print(generated_text)
        print("==========================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TinyLLM interface")
    parser.add_argument('--model', type=str, default='tinyllm_model.pth', help='Path to trained model file')
    parser.add_argument('--mode', type=str, default='web', choices=['web', 'cli'], help='Interface mode: web or cli')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host for web interface')
    parser.add_argument('--port', type=int, default=5000, help='Port for web interface')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on (cpu or cuda)')
    args = parser.parse_args()
    
    # Load model
    load_model(args.model, args.device)
    
    if args.mode == 'web':
        # Create HTML template
        create_templates()
        
        # Run Flask app
        print(f"Starting web interface at http://{args.host}:{args.port}")
        app.run(host=args.host, port=args.port, debug=False)
    else:
        # Run CLI
        run_cli()
