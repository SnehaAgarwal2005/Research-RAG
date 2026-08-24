import os
import json
import numpy as np
from google import genai
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss


# ============================================================
# 1. Gemini API
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )

client = genai.Client(api_key=API_KEY)


# ============================================================
# 2. Load embedding model
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")


# ============================================================
# 3. Load chunks
# ============================================================

with open(
    "data/chunks/chunks.json",
    "r",
    encoding="utf-8"
) as file:
    chunks = json.load(file)

texts = [chunk["text"] for chunk in chunks]


# ============================================================
# 4. Load FAISS index
# ============================================================

index = faiss.read_index(
    "data/vector_db/index.faiss"
)


# ============================================================
# 5. Create BM25 index
# ============================================================

tokenized_texts = [
    text.lower().split()
    for text in texts
]

bm25 = BM25Okapi(tokenized_texts)


# ============================================================
# 6. Function to answer one question
# ============================================================

def answer_question(query):

    # --------------------------------------------------------
    # FAISS semantic search
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        5
    )

    faiss_results = []

    for idx in indices[0]:

        if idx < len(chunks):
            faiss_results.append(chunks[idx])


    # --------------------------------------------------------
    # BM25 keyword search
    # --------------------------------------------------------

    bm25_scores = bm25.get_scores(
        query.lower().split()
    )

    top_bm25_indices = np.argsort(
        bm25_scores
    )[::-1][:5]

    bm25_results = []

    for idx in top_bm25_indices:
        bm25_results.append(chunks[idx])


    # --------------------------------------------------------
    # Hybrid RRF ranking
    # --------------------------------------------------------

    rrf_scores = {}

    k = 60


    # FAISS results

    for rank, chunk in enumerate(faiss_results):

        chunk_id = chunk["chunk_id"]

        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0

        rrf_scores[chunk_id] += (
            1 / (k + rank + 1)
        )


    # BM25 results

    for rank, chunk in enumerate(bm25_results):

        chunk_id = chunk["chunk_id"]

        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = 0

        rrf_scores[chunk_id] += (
            1 / (k + rank + 1)
        )


    # --------------------------------------------------------
    # Sort by RRF score
    # --------------------------------------------------------

    chunk_lookup = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    ranked_chunk_ids = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )


    # Top 5 chunks

    combined = [
        chunk_lookup[chunk_id]
        for chunk_id in ranked_chunk_ids[:5]
    ]


    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = "\n\n".join(
        [
            f"Source: {chunk['paper']} | "
            f"Page: {chunk.get('page', 'unknown')}\n"
            f"{chunk['text']}"
            for chunk in combined
        ]
    )


    # --------------------------------------------------------
    # Gemini prompt
    # --------------------------------------------------------

    prompt = f"""
You are a helpful research assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:

"I could not find the answer in the provided documents."

Do not invent information.

Give a clear and concise answer.

## CONTEXT

{context}

## QUESTION

{query}
"""


    # --------------------------------------------------------
    # Gemini
    # --------------------------------------------------------

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )


    # --------------------------------------------------------
    # Return answer + sources
    # --------------------------------------------------------

    sources = []

    for chunk in combined:

        sources.append({
            "paper": chunk["paper"],
            "page": chunk.get(
                "page",
                "unknown"
            ),
            "chunk": chunk.get(
                "chunk_number",
                chunk["chunk_id"]
            )
        })


    return {
        "answer": response.text,
        "sources": sources
    }


# ============================================================
# 7. Load evaluation questions
# ============================================================

with open(
    "evaluation.json",
    "r",
    encoding="utf-8"
) as file:

    evaluation_questions = json.load(file)


# ============================================================
# 8. Run evaluation
# ============================================================

results = []


print("\n")
print("=" * 70)
print("STARTING RAG EVALUATION")
print("=" * 70)


for number, item in enumerate(
    evaluation_questions,
    start=1
):

    question = item["question"]
    expected = item["expected"]


    print("\n")
    print("=" * 70)
    print(f"QUESTION {number}")
    print("=" * 70)

    print(question)


    result = answer_question(question)


    print("\nANSWER:")
    print(result["answer"])


    print("\nSOURCES:")

    for source in result["sources"]:

        print(
            f"- {source['paper']} | "
            f"Page {source['page']} | "
            f"Chunk {source['chunk']}"
        )


    results.append({

        "question": question,

        "expected_answer": expected,

        "generated_answer": result["answer"],

        "sources": result["sources"]

    })


# ============================================================
# 9. Save results
# ============================================================

with open(
    "evaluation_results.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4,
        ensure_ascii=False
    )


print("\n")
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print("evaluation_results.json")