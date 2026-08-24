import streamlit as st
import requests
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Research RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"

# ============================================================
# HEADER
# ============================================================

st.title("📚 Research RAG Assistant")

st.markdown(
    """
    **Research-paper question answering powered by Hybrid Retrieval + Gemini**

    Ask questions about your research papers using:

    **FAISS + BM25 + Reciprocal Rank Fusion (RRF) + Gemini**
    """
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Research Papers")

    uploaded_files = st.file_uploader(
        "Upload research papers",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        os.makedirs("data/raw_papers", exist_ok=True)

        for uploaded_file in uploaded_files:

            file_path = os.path.join(
                "data/raw_papers",
                uploaded_file.name
            )

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.success(
            f"{len(uploaded_files)} PDF(s) uploaded."
        )

    st.divider()

    # ========================================================
    # API HEALTH CHECK
    # ========================================================

    st.header("🔌 API Status")

    try:

        health_response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        if health_response.status_code == 200:

            st.success("🟢 FastAPI connected")

        else:

            st.warning("🟡 FastAPI responded with an error")

    except requests.exceptions.RequestException:

        st.error("🔴 FastAPI is not running")

    st.divider()

    # ========================================================
    # RETRIEVAL PIPELINE
    # ========================================================

    st.header("🔎 Retrieval Pipeline")

    st.markdown(
        """
        **1.** BGE Embeddings

        **2.** FAISS Semantic Search

        **3.** BM25 Keyword Search

        **4.** Reciprocal Rank Fusion

        **5.** Gemini Generation
        """
    )

    st.divider()

    # ========================================================
    # PROJECT INFORMATION
    # ========================================================

    st.header("📊 Project")

    st.write("📄 Research papers: 4")
    st.write("🧩 Indexed chunks: 178")
    st.write("🔍 Hybrid retrieval: Enabled")
    st.write("🤖 Gemini: Enabled")

# ============================================================
# MAIN QUESTION AREA
# ============================================================

st.subheader("Ask a question about your research papers")

question = st.text_input(
    "Question",
    placeholder="e.g. How does QLoRA reduce memory usage?",
    label_visibility="collapsed"
)

# ============================================================
# ASK BUTTON
# ============================================================

if st.button(
    "🔍 Ask",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "🔎 Searching papers and generating answer..."
        ):

            try:

                # =================================================
                # CALL FASTAPI
                # =================================================

                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question.strip()
                    },
                    timeout=180
                )

                # =================================================
                # SUCCESS
                # =================================================

                if response.status_code == 200:

                    result = response.json()

                    # =================================================
                    # ANSWER
                    # =================================================

                    st.divider()

                    st.subheader("🤖 Answer")

                    answer = result.get(
                        "answer",
                        "No answer was returned."
                    )

                    st.markdown(answer)

                    # =================================================
                    # SOURCES
                    # =================================================

                    st.subheader("📚 Sources")

                    sources = result.get(
                        "sources",
                        []
                    )

                    if sources:

                        for index, source in enumerate(
                            sources,
                            start=1
                        ):

                            paper = source.get(
                                "paper",
                                "Unknown paper"
                            )

                            page = source.get(
                                "page",
                                "?"
                            )

                            chunk = source.get(
                                "chunk",
                                "?"
                            )

                            st.markdown(
                                f"""
                                **{index}.** `{paper}`  
                                📄 Page: **{page}**  
                                🧩 Chunk: **{chunk}**
                                """
                            )

                    else:

                        st.info(
                            "No source information was returned."
                        )

                    # =================================================
                    # PIPELINE DETAILS
                    # =================================================

                    with st.expander(
                        "🔧 View RAG Pipeline"
                    ):

                        st.markdown(
                            """
                            ### Retrieval

                            **Embedding Model**

                            `BAAI/bge-small-en-v1.5`

                            **Semantic Search**

                            FAISS

                            **Keyword Search**

                            BM25

                            **Hybrid Ranking**

                            Reciprocal Rank Fusion (RRF)

                            **Generation**

                            Gemini

                            **Retrieved Context**

                            Top-ranked research-paper chunks
                            """
                        )

                # =================================================
                # API ERROR
                # =================================================

                else:

                    st.error(
                        f"❌ API request failed "
                        f"(HTTP {response.status_code})"
                    )

                    with st.expander(
                        "View API response"
                    ):

                        st.code(
                            response.text
                        )

            # =====================================================
            # CONNECTION ERROR
            # =====================================================

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to the FastAPI server."
                )

                st.info(
                    """
                    Make sure Terminal 1 is running:

                    `python -m uvicorn api.main:app --reload`
                    """
                )

            # =====================================================
            # TIMEOUT
            # =====================================================

            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ The request took too long."
                )

                st.info(
                    "Please try the question again."
                )

            # =====================================================
            # OTHER ERROR
            # =====================================================

            except Exception as e:

                st.error(
                    "❌ An unexpected error occurred."
                )

                with st.expander(
                    "Technical details"
                ):

                    st.code(
                        str(e)
                    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Research RAG Assistant • "
    "FAISS + BM25 + RRF + Gemini"
)