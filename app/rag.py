from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.llm import client
from app.vector_store import hybrid_search
from app.reranker import rerank_chunks


def chunk_document(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_text(text)

def ask_question(
            question,
            index,
            chunks,
            metadata,
            bm25,
            chat_history
):
    retrieved_chunks = hybrid_search(
        question,
        index,
        chunks,
        metadata,
        bm25,
        top_k=15
    )

    retrieved_chunks = rerank_chunks(
        question,
        retrieved_chunks,
        top_k=5
    )

    print("\n===== RETRIEVED CHUNKS =====")

    for chunk in retrieved_chunks:

        print(chunk)
        print("=" * 50)

    context = "\n\n".join(
        [
            item["text"]
            for item in retrieved_chunks
        ]
    )

    sources = []

    for item in retrieved_chunks:

        document_name = item["document"]

        if document_name not in sources:
            sources.append(document_name)

    history_text = ""

    for item in chat_history[-5:]:
        history_text += (
            f"User: {item['question']}\n"
            f"Assistant: {item['answer']}\n\n"
        )
    prompt = f"""
You are a financial document assistant.

Use ONLY the information present in the context.

Rules:
- Do not make up information.
- If information is unavailable, say:
  "The information was not found in the uploaded documents."
- Quote exact invoice numbers, dates and amounts when available.
- If multiple documents contain relevant information, summarize all of them.


Conversation History:
{history_text}

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content

    source_text = "\n".join(
        [
            f"• {doc}"
            for doc in sources
        ]
    )

    return f"""
{answer}

Sources:
{source_text}
"""