from sentence_transformers import CrossEncoder

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank_chunks(
        question,
        retrieved_chunks,
        top_k=5
):

    pairs = []

    for item in retrieved_chunks:

        pairs.append(
            (
                question,
                item["text"]
            )
        )

    scores = reranker.predict(pairs)

    scored_results = []

    for item, score in zip(
            retrieved_chunks,
            scores
    ):
        scored_results.append(
            (score, item)
        )

    scored_results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        item
        for score, item in scored_results[:top_k]
    ]