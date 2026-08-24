import json
from sentence_transformers import SentenceTransformer

# Load our embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Load our chunks
with open("data/chunks/chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print("Number of chunks:", len(chunks))

# Get the text from each chunk
texts = [chunk["text"] for chunk in chunks]

# Convert text into embeddings
embeddings = model.encode(
    texts,
    show_progress_bar=True
)

print("Embedding shape:", embeddings.shape)

# Save embeddings
with open("data/chunks/embeddings.json", "w", encoding="utf-8") as file:
    json.dump(embeddings.tolist(), file)

print("Embeddings saved successfully!")