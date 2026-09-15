"""
04_supervisor_agent.py
-----------------------
Topic: Supervisor Agent that routes tasks to worker agents (Gemini)
What:    Supervisor reads the question and picks one of three worker agents
Meaning: Single entry point — LLM classifies → correct specialist handles it
Example: "144 / 12?" → math_agent | "Reverse a string?" → code_agent | else → qa_agent
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)
parser = StrOutputParser()

# ── Worker agents ──────────────────────────────────────────────────────────────
math_agent = (
    ChatPromptTemplate.from_template("Solve this math problem step by step: {input}")
    | llm | parser
)

code_agent = (
    ChatPromptTemplate.from_template("Write a short Python code snippet for: {input}")
    | llm | parser
)

qa_agent = (
    ChatPromptTemplate.from_template("Answer this general question: {input}")
    | llm | parser
)

# ── Supervisor ────────────────────────────────────────────────────────────────
supervisor_chain = (
    ChatPromptTemplate.from_template("""
You are a supervisor. Reply with exactly one word:
- "math"  → math/calculation question
- "code"  → coding/programming question
- "qa"    → everything else

Question: {input}
Answer (one word only):""")
    | llm | parser
)

# ── Router ────────────────────────────────────────────────────────────────────
def run(user_input: str) -> str:
    route = supervisor_chain.invoke({"input": user_input}).strip().lower()
    print(f"[Supervisor] → {route}")
    if route == "math":
        return math_agent.invoke({"input": user_input})
    elif route == "code":
        return code_agent.invoke({"input": user_input})
    else:
        return qa_agent.invoke({"input": user_input})

if __name__ == "__main__":
    questions = [
        "What is 144 divided by 12?",
        "Write a Python function to reverse a string.",
        "Who invented the telephone?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {run(q)}")
