import tiktoken

# Initialize the tokenizer for your specific model
tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')  # Replace with your model name

def count_tokens(text):
    # text = """
    # The quick brown fox jumps over the lazy dog.
    # """ # Put your text here

    print(text)
    tokens = tokenizer.encode(text)
    token_count = len(tokens)

    print(f"Token count: {token_count}")