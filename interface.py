from generate import generate

def main():
    print("=== Tiny LLM ===")
    while True:
        prompt = input("\nEnter a prompt (or 'quit' to exit): ")
        if prompt.lower() == 'quit':
            break
        output = generate(prompt)
        print("\nGenerated Text:\n")
        print(output)

if __name__ == "__main__":
    main()
