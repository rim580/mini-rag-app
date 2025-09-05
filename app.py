import streamlit as st
import time
from rag_utils import get_embedder, split_into_chunks, retrieve_top_k, build_context, generate_answer, estimate_tokens

st.title("Mini RAG Demo (Free LLM)")

# Upload or paste
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
user_text = st.text_area("Or paste your text here")

query = st.text_input("Ask a question:")

if st.button("Get Answer"):
    start_time = time.time()
    embedder = get_embedder()

    # Get text from upload or paste
    text = ""
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
    elif user_text:
        text = user_text

    if not text.strip():
        st.warning("Please upload or paste some text.")
    else:
        # Split into chunks
        chunks = split_into_chunks(text)
        hits = retrieve_top_k(query, chunks, embedder)
        context = build_context(chunks, hits)

        # Generate answer
        answer = generate_answer(query, context)
        token_cost = estimate_tokens(query + context + answer)

        end_time = time.time()

        # Display
        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        st.write([chunks[i] for i, _ in hits])

        st.subheader("Request Info")
        st.write(f"Time taken: {end_time - start_time:.2f} sec")
        st.write(f"Estimated tokens: {token_cost}")
