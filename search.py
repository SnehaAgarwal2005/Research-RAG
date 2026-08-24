import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Load FAISS index
index = faiss.read_index("data/vector_db/index.faiss")

# Load chunks
with open("data/chunks/chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

# Ask the user a question
query = input("\nEnter your research question: ")

# Convert the question into an embedding
query_embedding = model.encode([query])

# Convert to NumPy float32
query_embedding = np.array(query_embedding).astype("float32")

# Search for the 5 closest chunks
distances, indices = index.search(query_embedding, 10)

print("\n")
print("=" * 70)
print("TOP 5 RETRIEVED CHUNKS")
print("=" * 70)

for rank in range(10):

    index_number = indices[0][rank]
    distance = distances[0][rank]

    chunk = chunks[index_number]

    print(f"\nRESULT {rank + 1}")
    print("-" * 70)

    print("Paper:", chunk["paper"])
    print("Page:", chunk["page"])
    print("Chunk:", chunk["chunk_number"])
    print("Distance:", round(float(distance), 4))

    print("\nText:")
    print(chunk["text"][:800])

print("\n" + "=" * 70)
print("SEARCH COMPLETE")
print("=" * 70)