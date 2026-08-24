import os
import json
import re
import numpy as np
import pymupdf
import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

PDF_FOLDER = "data/raw_papers"
CHUNK_FOLDER = "data/chunks"
INDEX_FOLDER = "data/vector_db"

CHUNK_SIZE = 350
OVERLAP = 50

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(CHUNK_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


# ============================================================
# CHUNK FUNCTION
# ============================================================

def split_into_chunks(text):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + CHUNK_SIZE

        chunk_words = words[start:end]

        if chunk_words:
            chunks.append(" ".join(chunk_words))

        start = end - OVERLAP

    return chunks


# ============================================================
# PROCESS PDFs
# ============================================================

all_chunks = []

chunk_id = 0

pdf_files = [
    f for f in os.listdir(PDF_FOLDER)
    if f.lower().endswith(".pdf")
]

print(f"Found {len(pdf_files)} PDF files.")


for filename in pdf_files:

    pdf_path = os.path.join(
        PDF_FOLDER,
        filename
    )

    print("Processing:", filename)

    document = pymupdf.open(pdf_path)

    for page_number, page in enumerate(
        document,
        start=1
    ):

        page_text = page.get_text()

        if not page_text.strip():
            continue

        page_chunks = split_into_chunks(
            page_text
        )

        for chunk_number, chunk_text in enumerate(
            page_chunks,
            start=1
        ):

            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "paper": filename,
                    "page": page_number,
                    "chunk_number": chunk_number,
                    "text": chunk_text
                }
            )

            chunk_id += 1

    document.close()


# ============================================================
# SAVE CHUNKS
# ============================================================

chunks_path = os.path.join(
    CHUNK_FOLDER,
    "chunks.json"
)

with open(
    chunks_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_chunks,
        file,
        indent=2,
        ensure_ascii=False
    )


print(
    f"Created {len(all_chunks)} chunks."
)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

texts = [
    chunk["text"]
    for chunk in all_chunks
]

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.asarray(
    embeddings,
    dtype="float32"
)


# ============================================================
# CREATE FAISS INDEX
# ============================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(embeddings)


# ============================================================
# SAVE FAISS INDEX
# ============================================================

index_path = os.path.join(
    INDEX_FOLDER,
    "index.faiss"
)

faiss.write_index(
    index,
    index_path
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 60)
print("INDEXING COMPLETE")
print("=" * 60)
print("PDFs:", len(pdf_files))
print("Chunks:", len(all_chunks))
print("Embeddings:", len(embeddings))
print("FAISS index:", index_path)
print("=" * 60)