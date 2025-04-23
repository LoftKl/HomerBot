import csv

# Input and output file paths
input_file = 'archive(1)/simpsons_script_lines.csv'     # Change this to your actual filename
output_file = 'homer_quotes.csv'

homer_lines = []

with open(input_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['raw_character_text'] == 'Homer Simpson' and row['speaking_line'].lower() == 'true':
            homer_lines.append({'spoken_words': row['spoken_words']})

# This part ensures proper CSV formatting, including commas and quotes
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['spoken_words'], quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(homer_lines)

print(f"Saved {len(homer_lines)} Homer Simpson quotes to '{output_file}'")
