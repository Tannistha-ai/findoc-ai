# test_vector.py

from app.ocr import extract_text
from app.rag import chunk_document
from app.vector_store import (
    create_vector_store,
    search_chunks
)

text = extract_text(
    "sample_docs/sample.pdf"
)

chunks = chunk_document(text)

index, stored_chunks = create_vector_store(
    chunks
)

results = search_chunks(
    "What bank account should payment be sent to?",
    index,
    stored_chunks
)

for r in results:
    print(r)
    print("=" * 50)