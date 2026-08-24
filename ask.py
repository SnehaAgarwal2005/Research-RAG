import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load embedding model
# -----------------------------
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# -----------------------------
# Load chunks
# -----------------------------
with open("data/chunks/chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

# -----------------------------
# Load FAISS index
# -----------------------------
index = faiss.read_index("data/vector_db/index.faiss")

# -----------------------------
# Ask a question
# -----------------------------
question = input("\nAsk your research question: ")

# Convert question to embedding
query_embedding = model.encode(
    [question],
    normalize_embeddings=True
).astype("float32")

# Search top 5 chunks
distances, indices = index.search(query_embedding, 5)

print("\n" + "=" * 70)
print("RETRIEVED EVIDENCE")
print("=" * 70)

for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), 1):

    chunk = chunks[idx]

    print(f"\nRESULT {rank}")
    print("-" * 70)
    print(f"Paper: {chunk['paper']}")
    print(f"Page: {chunk.get('page', 'N/A')}")
    print(f"Chunk: {chunk.get('chunk_number', 'N/A')}")
    print(f"Distance: {distance:.4f}")

    print("\nText:")
    print(chunk["text"][:2500])

print("\n" + "=" * 70)
print("RETRIEVAL COMPLETE")
print("=" * 70)