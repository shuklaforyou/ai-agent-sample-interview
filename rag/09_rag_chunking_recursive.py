"""
09_rag_chunking_recursive.py
-----------------------------
Topic: RAG - Recursive Character Text Splitting (most common in practice)
What:    Tries to split on \n\n, then \n, then space — largest boundary first
Meaning: Keeps paragraphs/sentences intact; only splits words as a last resort
Example: 3 paragraphs → [Chunk1: full para] [Chunk2: partial with overlap]
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Sample document ---
text = """
Artificial Intelligence (AI) is the simulation of human intelligence by machines.

Machine Learning (ML) is a subset of AI. ML algorithms learn from data to make predictions.

Deep Learning is a subset of ML. It uses neural networks with many layers.

Large Language Models (LLMs) are deep learning models trained on massive text datasets.
They can generate text, answer questions, translate languages, and write code.
"""

# --- Recursive splitter: tries \n\n → \n → " " → "" in order ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_text(text)

if __name__ == "__main__":
    print(f"Total chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---\n{chunk}\n")
