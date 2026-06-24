from rank_bm25 import BM25Okapi


def create_bm25(chunks):

    tokenized_chunks = [
        chunk.split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(
        tokenized_chunks
    )

    return bm25


def bm25_search(
        query,
        bm25,
        chunks,
        metadata,
        top_k=5
):

    tokenized_query = query.split()

    scores = bm25.get_scores(
        tokenized_query
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []

    for idx in ranked_indices:

        results.append(
            {
                "text": chunks[idx],
                "document": metadata[idx]["document"]
            }
        )

    return results