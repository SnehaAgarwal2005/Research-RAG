import json
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading chunks...")

with open("data/chunks/chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print("Total chunks:", len(chunks))


# ============================================================
# 2. LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")


# ============================================================
# 3. LOAD FAISS INDEX
# ============================================================

print("Loading FAISS index...")

index = faiss.read_index("data/vector_db/index.faiss")


# ============================================================
# 4. CREATE BM25 INDEX
# ============================================================

print("Creating BM25 index...")

documents = []

for chunk in chunks:
    text = chunk["text"]

    # Simple tokenization
    tokens = text.lower().split()

    documents.append(tokens)

bm25 = BM25Okapi(documents)

print("BM25 index ready!")


# ============================================================
# 5. ASK QUESTION
# ============================================================

question = input("\nEnter your research question: ")


# ============================================================
# 6. FAISS SEARCH
# ============================================================

query_embedding = model.encode(
    [question],
    normalize_embeddings=True
).astype("float32")

faiss_distances, faiss_indices = index.search(
    query_embedding,
    10
)


# ============================================================
# 7. BM25 SEARCH
# ============================================================

query_tokens = question.lower().split()

bm25_scores = bm25.get_scores(query_tokens)

# Get top 10 BM25 results
bm25_indices = np.argsort(bm25_scores)[::-1][:10]


# ============================================================
# 8. COMBINE RESULTS
# ============================================================

# Store scores for each chunk
combined_scores = {}


# -----------------------------
# FAISS scores
# -----------------------------

for rank, idx in enumerate(faiss_indices[0]):

    # Higher rank = higher score
    score = 1 / (rank + 1)

    combined_scores[int(idx)] = combined_scores.get(int(idx), 0) + score


# -----------------------------
# BM25 scores
# -----------------------------

for rank, idx in enumerate(bm25_indices):

    score = 1 / (rank + 1)

    combined_scores[int(idx)] = combined_scores.get(int(idx), 0) + score


# Sort combined results
final_results = sorted(
    combined_scores.items(),
    key=lambda x: x[1],
    reverse=True
)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("HYBRID SEARCH RESULTS")
print("=" * 70)


for rank, (idx, score) in enumerate(final_results[:10], 1):

    chunk = chunks[idx]

    print(f"\nRESULT {rank}")
    print("-" * 70)

    print("Paper:", chunk["paper"])
    print("Page:", chunk.get("page", "N/A"))
    print("Chunk:", chunk.get("chunk_number", "N/A"))
    print("Combined score:", round(score, 4))

    print("\nText:")
    print(chunk["text"][:1500])


print("\n" + "=" * 70)
print("HYBRID SEARCH COMPLETE")
print("=" * 70)