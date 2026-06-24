import streamlit as st
import tempfile
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from app.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store
)
from app.bm25_retriever import create_bm25

from app.ocr import extract_text
from app.llm import (
    summarize_financial_text,
    extract_financial_fields,
    analyze_invoice
)
from app.rag import ask_question, chunk_document


st.set_page_config(
    page_title="FinDoc AI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 FinDoc AI")
st.subheader("AI-Powered Financial Document Analyzer")

# ---------------- SESSION STATE ----------------

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "documents" not in st.session_state:
    st.session_state.documents = []

if "all_invoices" not in st.session_state:
    st.session_state.all_invoices = []

if "all_amounts" not in st.session_state:
    st.session_state.all_amounts = []

if "global_index" not in st.session_state:

    loaded_index, loaded_chunks, loaded_metadata = load_vector_store()

    if loaded_index is not None:
        st.session_state.global_index = loaded_index
        st.session_state.global_chunks = loaded_chunks
        st.session_state.global_metadata = loaded_metadata

        st.session_state.global_bm25 = create_bm25(
            loaded_chunks
        )
        st.session_state.analysis_done = True

        st.success(
            "✅ Existing vector database loaded"
        )
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- FILE UPLOAD ----------------

uploaded_files = st.file_uploader(
    "Upload Financial Documents",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# ---------------- ANALYSIS ----------------

if uploaded_files:

    st.success(f"{len(uploaded_files)} file(s) uploaded")

    if st.button("Analyze Documents"):

        documents = []
        all_invoices = []
        all_amounts = []
        all_chunks = []
        all_metadata = []

        for uploaded_file in uploaded_files:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(uploaded_file.name)[1]
            ) as tmp_file:

                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

            extracted_text = extract_text(temp_path)

            chunks = chunk_document(extracted_text)
            #individual document RAG
            index, stored_chunks = create_vector_store(
                chunks
            )
            bm25 = create_bm25(chunks)


            doc_metadata = []

            for _ in stored_chunks:
                doc_metadata.append(
                    {
                        "document": uploaded_file.name
                    }
                )

            #FOR GLOBAL RAG
            for chunk in chunks:
                all_chunks.append(chunk)

                all_metadata.append(
                    {
                        "document": uploaded_file.name
                    }
                )


            summary = summarize_financial_text(extracted_text)

            fields = extract_financial_fields(extracted_text)

            analysis = analyze_invoice(extracted_text)

            try:

                fields_dict = json.loads(
                    fields.replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

                all_invoices.append(fields_dict)

                amount = (
                    str(fields_dict.get("total_amount", "0"))
                    .replace("$", "")
                    .replace(",", "")
                )

                try:
                    all_amounts.append(float(amount))
                except:
                    pass

            except:
                pass



            documents.append({
                "name": uploaded_file.name,
                "summary": summary,
                "fields": fields,
                "analysis": analysis,
                "index": index,
                "bm25": bm25,
                "chunks": stored_chunks,
                "metadata": doc_metadata
            })

        global_index, global_chunks = create_vector_store(
                all_chunks
        )
        global_bm25 = create_bm25(all_chunks)
        save_vector_store(
            global_index,
            all_chunks,
            all_metadata
        )

        st.session_state.documents = documents
        st.session_state.all_invoices = all_invoices
        st.session_state.all_amounts = all_amounts
        st.session_state.analysis_done = True

        st.session_state.global_index = global_index
        st.session_state.global_chunks = all_chunks
        st.session_state.global_metadata = all_metadata
        st.session_state.global_bm25 = global_bm25


# ---------------- DISPLAY RESULTS ----------------

if st.session_state.analysis_done:

    for doc in st.session_state.documents:

        st.divider()

        st.header(f"📄 {doc['name']}")

        st.subheader("🤖 AI Summary")
        st.markdown(doc["summary"])

        st.subheader("📊 Structured Data")

        try:

            fields_dict = json.loads(
                doc["fields"]
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

            st.json(fields_dict)

        except:
            st.code(doc["fields"])

        st.subheader("📈 Risk Analysis")

        try:

            risk_data = json.loads(
                doc["analysis"]
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Risk Score",
                f"{risk_data.get('risk_score', 0)}/100"
            )

            col2.metric(
                "Risk Level",
                risk_data.get("risk_level", "Unknown")
            )

            col3.metric(
                "Payment Status",
                risk_data.get("payment_status", "Unknown")
            )

            risk_level = risk_data.get(
                "risk_level",
                ""
            ).upper()

            if risk_level == "LOW":
                st.success("🟢 Low Risk Invoice")

            elif risk_level == "MEDIUM":
                st.warning("🟡 Medium Risk Invoice")

            else:
                st.error("🔴 High Risk Invoice")

            st.json(risk_data)

        except Exception:

            st.code(doc["analysis"])


        st.subheader("💬 Ask Questions")

        question = st.text_input(
            "Ask a question",
            key=f"question_{doc['name']}"
        )

        if st.button(
                "Ask",
                key=f"ask_{doc['name']}"
        ):

            if question.strip():
                answer = ask_question(
                    question,
                    doc["index"],
                    doc["chunks"],
                    doc["metadata"],
                    doc["bm25"],
                    st.session_state.chat_history
                )

                clean_answer = answer.split("Sources:")[0]

                st.session_state.chat_history.append(
                    {
                        "question": question,
                        "answer": clean_answer
                    }
                )

                st.success(answer)


    # ---------------- DASHBOARD ----------------

    all_invoices = st.session_state.all_invoices
    all_amounts = st.session_state.all_amounts

    st.divider()
    st.header("📈 Multi-Document Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Documents", len(all_invoices))
    col2.metric("Total Value", f"${sum(all_amounts):,.2f}")
    average = (
        sum(all_amounts) / len(all_amounts)
        if all_amounts else 0
    )

    col3.metric(
        "Average Value",
        f"${average:,.2f}"
    )

    if all_invoices:

        df = pd.DataFrame(all_invoices)

        st.subheader("📋 Invoice Portfolio")
        st.dataframe(df)

        csv = df.to_csv(index=False)

        st.download_button(
            "⬇ Download CSV",
            csv,
            "invoice_report.csv",
            "text/csv"
        )

        try:

            chart_df = df.copy()

            chart_df["total_amount"] = (
                chart_df["total_amount"]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .astype(float)
            )

            fig, ax = plt.subplots()

            ax.bar(
                chart_df["invoice_number"],
                chart_df["total_amount"]
            )

            st.pyplot(fig)

        except:
            pass
st.divider()

st.header("🔍 Ask Across All Documents")

global_question = st.text_input(
    "Ask a question across all uploaded documents",
    key="global_question"
)

if (
    st.session_state.analysis_done
    and st.button("Search All Documents")
):

    if global_question.strip():
        answer = ask_question(
            global_question,
            st.session_state.global_index,
            st.session_state.global_chunks,
            st.session_state.global_metadata,
            st.session_state.global_bm25,
            st.session_state.chat_history
        )

        answer = ask_question(
            global_question,
            st.session_state.global_index,
            st.session_state.global_chunks,
            st.session_state.global_metadata,
            st.session_state.global_bm25,
            st.session_state.chat_history
        )
        clean_answer = answer.split("Sources:")[0]

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": clean_answer
            }
        )

        st.success(answer)
if st.session_state.chat_history:

    st.divider()
    st.header("💬 Conversation History")

    for item in reversed(
        st.session_state.chat_history
    ):
        st.markdown(
            f"**Q:** {item['question']}"
        )

        st.markdown(
            f"**A:** {item['answer']}"
        )

        st.divider()