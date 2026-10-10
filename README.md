
In-Memory Retrieval-Augmented Generation (RAG) Pipeline

This code provides a lightweight, local, in-memory RAG pipeline that chunks raw text, retrieves relevant passages using semantic similarity, and generates answers using Hugging Face models.

Overview

The pipeline uses sentence-transformers for embedding generation, vector dot-product for dense retrieval, and a Hugging Face text2text-generation model to answer user questions strictly based on retrieved context.

Key Components

• estimate_tokens(text): Approximates token count using a standard character-to-token ratio heuristic (~4 characters per token).
• get_embedder() / get_generator(): Implements lazy singleton loading for all-MiniLM-L6-v2 (dense vector embeddings) and google/flan-t5-small (generation).
• split_into_chunks(text, chunk_size=800, overlap=120): Splices input documents into overlapping sliding-window word chunks to preserve context across boundaries.
• embed_texts(embedder, texts): Encodes text segments into L₂-normalized vector representations (float32).
• retrieve_top_k(query, chunks, embedder, k=4): Calculates cosine similarity via matrix dot product (emb_matrix @ q) and returns the top k ranked chunk indices with their scores.
• build_context(chunks, hits, max_chars=3000): Compiles retrieved passages into an aggregated context string while respecting a character budget.
• generate_answer(query, context, max_new_tokens=200): Formulates an instruction prompt and executes greedy decoding (do_sample=False) using Flan-T5 to return the answer.

Dependencies & Installation

Install the required Python packages:
bash
pip install numpy sentence-transformers transformers torch


Usage Example

python
# Sample document
document = """
Transformers are deep learning models introduced in 2017. 
They use self-attention mechanisms to process sequential data in parallel. 
"""

# 1. Chunk document
chunks = split_into_chunks(document, chunk_size=50, overlap=10)

# 2. Retrieve top chunks
embedder = get_embedder()
query = "Why are transformers faster than RNNs?"
hits = retrieve_top_k(query, chunks, embedder, k=2)

# 3. Assemble context and generate answer
context = build_context(chunks, hits)
answer = generate_answer(query, context)

