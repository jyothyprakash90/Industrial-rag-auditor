import streamlit as st
from app.ingest import ingest_pdf
from app.rag_pipeline import answer_question
from app import config

st.set_page_config(page_title="Industrial RAG Auditor", page_icon="🏭")
st.title("🏭 Industrial RAG Auditor")
st.caption("Upload an audit report and ask compliance questions grounded in its content.")

if not config.OPENAI_API_KEY:
    st.error(
        "No OpenAI API key found. Add OPENAI_API_KEY to your .env file "
        "(see .env.example)."
    )
    st.stop()

# Persist the vectorstore across reruns within a session so we don't
# re-embed the PDF every time the user asks a new question.
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    st.session_state.filename = None

uploaded_file = st.file_uploader("Upload an audit report (PDF)", type=["pdf"])

if uploaded_file is not None and uploaded_file.name != st.session_state.filename:
    with st.spinner("Reading and indexing report..."):
        try:
            vectorstore, num_chunks = ingest_pdf(uploaded_file)
            st.session_state.vectorstore = vectorstore
            st.session_state.filename = uploaded_file.name
            st.success(f"Indexed {num_chunks} chunks from {uploaded_file.name}.")
        except ValueError as e:
            st.error(str(e))
            st.stop()

if st.session_state.vectorstore is not None:
    question = st.text_input(
        "Ask a compliance question about this report",
        placeholder='e.g. "Does this factory meet ISO 9001 standards based on these notes?"',
    )

    if question:
        with st.spinner("Retrieving relevant sections and generating answer..."):
            result = answer_question(st.session_state.vectorstore, question)

        st.subheader("Answer")
        st.write(result["answer"])

        with st.expander("View source chunks used"):
            for i, doc in enumerate(result["sources"], start=1):
                st.markdown(f"**Chunk {i}**")
                st.text(doc.page_content)
                st.divider()
