"""
10_rag_chunking_semantic.py
----------------------------
Topic: RAG - Semantic Chunking (splits at meaning boundaries)
What:    Uses embeddings to detect where the topic/meaning changes significantly
Meaning: Groups sentences that talk about the same idea into one chunk
Example: Eiffel Tower sentences + Python sentences → 2 separate chunks (different topics)
"""

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

# --- Embeddings model ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- Semantic chunker: splits where meaning changes significantly ---
splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",   # options: percentile, std_dev, interquartile
    breakpoint_threshold_amount=90,           # higher = fewer splits
)

# --- Sample document ---
text = """
The Eiffel Tower is located in Paris, France. It was built in 1889.
It is one of the most visited monuments in the world.

Python is a high-level programming language. It is known for its simple syntax.
Python is widely used in data science and machine learning.

The Amazon River is the largest river by discharge volume. It flows through Brazil.
"""

chunks = splitter.split_text(text)

if __name__ == "__main__":
    print(f"Total chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---\n{chunk}\n")
