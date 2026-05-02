import streamlit as st
from utils.rag_pipeline import create_rag_pipeline

st.set_page_config(page_title="RAG AI Assistant", layout="centered")

st.title("📄 AI Knowledge Assistant (RAG)")
st.write("Ask questions from your documents")

# Upload file
uploaded_file = st.file_uploader("Upload a TXT or PDF file", type=["txt", "pdf"])

if uploaded_file:
    file_path = f"data/{uploaded_file.name}"
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded successfully!")

    if "pipeline" not in st.session_state:
        st.session_state.pipeline = create_rag_pipeline(file_path)

    query = st.text_input("Ask your question:")

    if query:
        pipeline = st.session_state.pipeline

        if hasattr(pipeline, "run"):
            answer = pipeline.run(query)
        else:
            docs = pipeline.get_relevant_documents(query)
            answer = "\n\n".join([doc.page_content for doc in docs[:2]])

        st.subheader("🤖 Answer")
        st.write(answer)
