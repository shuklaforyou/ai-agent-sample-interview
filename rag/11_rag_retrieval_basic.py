"""
11_rag_retrieval_basic.py
--------------------------
Topic: RAG - Basic Similarity Retrieval with FAISS
What:    Embeds docs into vectors, stores in FAISS, retrieves top-k by cosine similarity
Meaning: "What is RAG?" → find the 2 most similar doc chunks → pass as context to LLM
Example: query → retriever.invoke(query) → [doc1, doc2] → LLM answers
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- 1. Sample documents ---
docs = [
    Document(page_content="LangChain helps build LLM applications with chains and agents."),
    Document(page_content="FAISS is a fast vector similarity search library by Meta."),
    Document(page_content="RAG combines retrieval with generation to answer questions."),
    Document(page_content="Embeddings convert text into numerical vectors for search."),
]

# --- 2. Embed and store in FAISS vector store ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(docs, embeddings)

# --- 3. Retriever: top-k similar docs ---
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- 4. Retrieve and answer ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def ask(query: str) -> str:
    relevant_docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in relevant_docs])
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    return llm.invoke(prompt).content

# --- 5. Run ---
if __name__ == "__main__":
    question = "What is RAG?"
    print(f"Q: {question}")
    print(f"A: {ask(question)}")
