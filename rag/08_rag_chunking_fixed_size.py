"""
08_rag_chunking_fixed_size.py
------------------------------
Topic: RAG - Fixed Size Chunking
What:    Splits text into chunks of a fixed character size (e.g. 150 chars)
Meaning: Ensures no chunk is too large for the embedding model to process
Example: "LangChain is a..." → [Chunk1: 150 chars] [Chunk2: 150 chars] with 30-char overlap
"""

from langchain.text_splitter import CharacterTextSplitter

# --- Sample document ---
text = """
LangChain is a framework designed to simplify the creation of applications using large language models.
It provides tools for chaining LLM calls, managing memory, and building agents.
RAG stands for Retrieval-Augmented Generation, a pattern that combines search with generation.
In RAG, documents are split into chunks, embedded, stored in a vector database, and retrieved at query time.
"""

# --- Fixed size splitter ---
splitter = CharacterTextSplitter(
    chunk_size=150,       # max characters per chunk
    chunk_overlap=30,     # overlap to preserve context
    separator="\n"
)

chunks = splitter.split_text(text)

if __name__ == "__main__":
    print(f"Total chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---\n{chunk}\n")
