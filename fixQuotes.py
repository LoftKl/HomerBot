"""
This wscript adds quotes to the beginning and end of each line in a text file,
unless the line already starts with a quote.

To use this wscript:
1.  Save it as a .py file (e.g., add_quotes.py).
2.  Open a command prompt or terminal.
3.  Navigate to the directory where you saved the file.
4.  Run the script using: `python add_quotes.py --file=<input_file_path> --output=<output_file_path>`

    * Replace `<input_file_path>` with the path to your input text file.
    * Replace `<output_file_path>` with the path where you want to save the output.
        If the output file exists, it will be overwritten.

For example:
    python add_quotes.py --file=input.txt --output=output.txt
"""
import argparse
import sys
import os

def add_quotes_to_lines(text):
    """
    Adds quotes to the beginning and end of each line in a text,
    unless the line already starts with a quote.

    Args:
        text: The input text as a string.

    Returns:
        The modified text with quotes added as a string.
    """
    lines = text.splitlines()
    modified_lines = []
    for line in lines:
        # Strip leading/trailing whitespace for accurate checking and cleaner output
        stripped_line = line.strip()
        if not stripped_line.startswith('"'):
            modified_lines.append(f'"{stripped_line}"')
        else:
            modified_lines.append(stripped_line)  # Keep original if already quoted
    return "\n".join(modified_lines)


def process_file(input_file_path, output_file_path):
    """
    Reads text from an input file, adds quotes to the lines, and writes the
    modified text to an output file.

    Args:
        input_file_path: Path to the input text file.
        output_file_path: Path to the output text file.
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            text = infile.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(1)

    modified_text = add_quotes_to_lines(text)

    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            outfile.write(modified_text)
        print(f"Processed file and saved to {output_file_path}")
    except Exception as e:
        print(f"Error writing to output file: {e}", file=sys.stderr)
        sys.exit(1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Add quotes to the beginning and end of lines in a text file.")
    parser.add_argument("--file", required=True, help="Path to the input text file.")
    parser.add_argument("--output", required=True, help="Path to the output text file.")

    args = parser.parse_args()
    input_file_path = args.file
    output_file_path = args.output

    # Check file paths
    if not os.path.exists(input_file_path):
        print(f"Error: Input file does not exist: {input_file_path}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(output_file_path):
        print(f"Overwriting existing output file: {output_file_path}")
    
    process_file(input_file_path, output_file_path)
    sys.exit(0) # Explicitly exit with success code
