"""
12_rag_retrieval_chain.py
--------------------------
Topic: RAG - Full Retrieval Chain using LCEL
What:    Chains retriever → prompt formatter → LLM → output parser using the | pipe syntax
Meaning: Clean, composable pipeline — each step transforms and passes output to the next
Example: question → retrieve docs → format context → fill prompt → LLM → plain string answer
"""

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. Documents ---
docs = [
    Document(page_content="The capital of France is Paris."),
    Document(page_content="Python was created by Guido van Rossum in 1991."),
    Document(page_content="The Great Wall of China is over 13,000 miles long."),
    Document(page_content="Albert Einstein developed the theory of relativity."),
]

# --- 2. Vector store ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- 3. Prompt ---
prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question.
If you don't know, say "I don't know."

Context: {context}

Question: {question}
""")

# --- 4. Format retrieved docs ---
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# --- 5. Full RAG chain using LCEL ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 6. Run ---
if __name__ == "__main__":
    questions = [
        "Who created Python?",
        "What is the capital of France?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {rag_chain.invoke(q)}")
