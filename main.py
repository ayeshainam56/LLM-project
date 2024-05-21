import open_clip
import torch
from transformers import CLIPTokenizer

# Load the OpenCLIP model and tokenizer
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def chunk_text(text, chunk_size):
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [" ".join(chunk) for chunk in chunks]

def embed_chunks(chunks, model, tokenizer, max_length):
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding='max_length', max_length=max_length)
        with torch.no_grad():
            outputs = model.encode_text(inputs.input_ids)
        embeddings.append(outputs)
    return embeddings

# Example usage
query = "Your complex query goes here. It might be long, so we need to chunk it."
chunk_size = 16  # Example chunk size
max_length = 77  # Maximum token length for the model
chunks = chunk_text(query, chunk_size)
embeddings = embed_chunks(chunks, model, tokenizer, max_length)

# Now you have a list of embeddings without combining them
for idx, embedding in enumerate(embeddings):
    print(f"Embedding for chunk {idx}:\n{embedding}\n")
