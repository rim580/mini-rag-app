import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# token estimator (rough heuristic)
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

# lazy load embedder
_embedder = None
def get_embedder(model_name="all-MiniLM-L6-v2"):
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(model_name)
    return _embedder

# lazy load generator
_generator = None
def get_generator(model_name="google/flan-t5-small"):
    global _generator
    if _generator is None:
        _generator = pipeline("text2text-generation", model=model_name)
    return _generator

def split_into_chunks(text: str, chunk_size: int = 800, overlap: int = 120):
    tokens = text.replace("\r\n", "\n").split()
    chunks, start = [], 0
    while start < len(tokens):
        end = min(len(tokens), start + chunk_size)
        chunks.append(" ".join(tokens[start:end]))
        if end == len(tokens): break
        start = end - overlap
        if start < 0: start = 0
    return chunks

def embed_texts(embedder, texts):
    embs = embedder.encode(texts, normalize_embeddings=True)
    return np.array(embs, dtype=np.float32)

def retrieve_top_k(query: str, chunks: list, embedder, k: int = 4):
    if not chunks:
        return []
    emb_matrix = embed_texts(embedder, chunks)
    q = embed_texts(embedder, [query])[0]
    scores = emb_matrix @ q
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]

def build_context(chunks: list, hits: list, max_chars: int = 3000):
    ctx = []
    total = 0
    for i, _s in hits:
        c = chunks[i]
        if total + len(c) <= max_chars:
            ctx.append(f"[Chunk {i}]\n{c}")
            total += len(c)
        else:
            break
    return "\n\n".join(ctx)

def generate_answer(query: str, context: str, max_new_tokens: int = 200):
    generator = get_generator()
    if not context.strip():
        return "I couldn’t find relevant content in the provided documents."

    prompt = f"Answer the question using only the context.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    result = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return result[0]["generated_text"].strip()
