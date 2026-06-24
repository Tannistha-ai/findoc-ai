from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

from app.bm25_retriever import bm25_search

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def create_vector_store(chunks):

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        np.array(embeddings).astype("float32")
    )

    return index, chunks


def search_chunks(
        query,
        index,
        chunks,
        metadata,
        top_k=15
):

    top_k = min(top_k, len(chunks))

    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        top_k
    )

    results = []

    for idx in indices[0]:

        results.append(
            {
                "text": chunks[idx],
                "document": metadata[idx]["document"]
            }
        )

    return results


def hybrid_search(
        query,
        index,
        chunks,
        metadata,
        bm25,
        top_k=10
):

    semantic_results = search_chunks(
        query,
        index,
        chunks,
        metadata,
        top_k
    )

    keyword_results = bm25_search(
        query,
        bm25,
        chunks,
        metadata,
        top_k
    )

    combined = []
    seen = set()

    for item in semantic_results + keyword_results:

        if item["text"] not in seen:

            combined.append(item)
            seen.add(item["text"])

    return combined[:top_k]


def save_vector_store(
        index,
        chunks,
        metadata,
        save_dir="vector_db"
):

    os.makedirs(save_dir, exist_ok=True)

    faiss.write_index(
        index,
        f"{save_dir}/faiss.index"
    )

    with open(
            f"{save_dir}/chunks.pkl",
            "wb"
    ) as f:

        pickle.dump(chunks, f)

    with open(
            f"{save_dir}/metadata.pkl",
            "wb"
    ) as f:

        pickle.dump(metadata, f)


def load_vector_store(
        save_dir="vector_db"
):

    index_path = f"{save_dir}/faiss.index"

    if not os.path.exists(index_path):
        return None, None, None

    index = faiss.read_index(index_path)

    with open(
            f"{save_dir}/chunks.pkl",
            "rb"
    ) as f:

        chunks = pickle.load(f)

    with open(
            f"{save_dir}/metadata.pkl",
            "rb"
    ) as f:

        metadata = pickle.load(f)

    return index, chunks, metadata