from app.ocr import extract_text
from app.rag import chunk_document

text = extract_text("sample_docs/sample.pdf")

chunks = chunk_document(text)

print("Total Chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {i + 1} ---")
    print(chunk)