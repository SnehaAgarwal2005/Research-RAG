import json
import numpy as np
import faiss
import os

# Load embeddings
with open("data/chunks/embeddings.json", "r", encoding="utf-8") as file:
    embeddings = json.load(file)

# Convert embeddings to NumPy array
embeddings = np.array(embeddings).astype("float32")

print("Embedding shape:", embeddings.shape)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

# Add embeddings to the index
index.add(embeddings)

print("Number of vectors in FAISS:", index.ntotal)

# Create output folder
os.makedirs("data/vector_db", exist_ok=True)

# Save FAISS index
faiss.write_index(index, "data/vector_db/index.faiss")

print("FAISS index saved successfully!")