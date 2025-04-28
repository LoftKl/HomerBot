def remove_quotations(input_file, output_file):
    # List of quotation characters to remove
    quotations = ["'", '"', '‘', '’', '“', '”']

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Remove all quotation characters
    for quote in quotations:
        text = text.replace(quote, '')

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)

if __name__ == "__main__":
    input_path = 'extracted_quotes2.csv'    # Replace with your file
    output_path = 'modelinput.txt'  # Replace with your desired output file
    remove_quotations(input_path, output_path)
    print(f"Quotations removed. Clean text saved to {output_path}")
