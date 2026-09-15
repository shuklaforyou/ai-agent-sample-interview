"""
13_rag_retrieval_mmr.py
------------------------
Topic: RAG - MMR (Maximal Marginal Relevance) Retrieval
What:    Retrieves docs that are relevant to the query AND different from each other
Meaning: Prevents returning 3 nearly identical Python docs when you asked about "AI languages"
Example: query → MMR picks Python doc + JS doc + Rust doc (diverse) vs 3x Python (redundant)
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# --- Documents (some are very similar intentionally) ---
docs = [
    Document(page_content="Python is a popular programming language used in AI."),
    Document(page_content="Python is widely used in machine learning and data science."),  # similar
    Document(page_content="JavaScript is used for building web applications."),
    Document(page_content="Rust is a systems programming language focused on safety."),
    Document(page_content="Data science involves statistics, programming, and domain knowledge."),
]

# --- Embed and store ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(docs, embeddings)

# --- MMR retriever: k=3 results, considers diversity with lambda_mult ---
# lambda_mult: 1.0 = pure relevance, 0.0 = pure diversity
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5}
)

# --- Regular similarity retriever for comparison ---
retriever_sim = vectorstore.as_retriever(search_kwargs={"k": 3})

if __name__ == "__main__":
    query = "programming languages for AI"

    print("=== Similarity Search (may have duplicates) ===")
    for doc in retriever_sim.invoke(query):
        print(f"  - {doc.page_content}")

    print("\n=== MMR Search (diverse results) ===")
    for doc in retriever_mmr.invoke(query):
        print(f"  - {doc.page_content}")
