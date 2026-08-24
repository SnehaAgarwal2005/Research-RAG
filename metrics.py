import json


# ============================================================
# Load evaluation results
# ============================================================

with open(
    "evaluation_results.json",
    "r",
    encoding="utf-8"
) as file:
    results = json.load(file)


# ============================================================
# Load evaluation questions
# ============================================================

with open(
    "evaluation.json",
    "r",
    encoding="utf-8"
) as file:
    evaluation = json.load(file)


# ============================================================
# Calculate Recall@5 and MRR
# ============================================================

recall_scores = []
mrr_scores = []


for item, result in zip(evaluation, results):

    expected_sources = set(
        item["expected_sources"]
    )

    retrieved_sources = [
        source["paper"]
        for source in result["sources"]
    ]

    # --------------------------------------------------------
    # Recall@5
    # --------------------------------------------------------

    found = any(
        source in expected_sources
        for source in retrieved_sources
    )

    recall = 1 if found else 0

    recall_scores.append(recall)


    # --------------------------------------------------------
    # MRR
    # --------------------------------------------------------

    reciprocal_rank = 0

    for rank, source in enumerate(
        retrieved_sources,
        start=1
    ):

        if source in expected_sources:

            reciprocal_rank = 1 / rank

            break

    mrr_scores.append(reciprocal_rank)


# ============================================================
# Overall metrics
# ============================================================

recall_at_5 = sum(recall_scores) / len(
    recall_scores
)

mrr = sum(mrr_scores) / len(
    mrr_scores
)


# ============================================================
# Display results
# ============================================================

print()
print("=" * 60)
print("RAG RETRIEVAL EVALUATION")
print("=" * 60)

print(
    f"Recall@5 : {recall_at_5:.2%}"
)

print(
    f"MRR      : {mrr:.3f}"
)

print()

for i, (recall, reciprocal_rank) in enumerate(
    zip(recall_scores, mrr_scores),
    start=1
):

    print(
        f"Question {i}: "
        f"Recall@5={recall} | "
        f"Reciprocal Rank={reciprocal_rank:.3f}"
    )

print("=" * 60)