from app.ocr import extract_text
from app.rag import chunk_document
from app.vector_store import create_vector_store
from app.rag import ask_question

text = extract_text("sample_docs/sample.pdf")

chunks = chunk_document(text)

index, stored_chunks = create_vector_store(chunks)

answer = ask_question(
    "What bank account should payment be sent to?",
    index,
    stored_chunks
)

print(answer)