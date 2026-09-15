"""
03_array_of_agents.py
----------------------
Topic: Array of Multiple Agents — Sequential Pipeline (Gemini)
What:    List of specialized agents, each runs on the output of the previous
Meaning: Pipeline pattern — same data flows through summarizer → translator → sentiment
Example: article → [Summarized] → [Translated to Hindi] → [Sentiment: Positive]
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

# ── Agent 1: Summarizer ───────────────────────────────────────────────────────
summarizer_agent = (
    ChatPromptTemplate.from_template("Summarize this in 2 sentences:\n\n{text}")
    | llm | parser
)

# ── Agent 2: Translator ───────────────────────────────────────────────────────
translator_agent = (
    ChatPromptTemplate.from_template("Translate to Hindi:\n\n{text}")
    | llm | parser
)

# ── Agent 3: Sentiment Analyzer ───────────────────────────────────────────────
sentiment_agent = (
    ChatPromptTemplate.from_template("Analyze the sentiment (positive/negative/neutral) of:\n\n{text}")
    | llm | parser
)

# ── Array of agents ───────────────────────────────────────────────────────────
agents = [
    ("Summarizer", summarizer_agent),
    ("Translator", translator_agent),
    ("Sentiment",  sentiment_agent),
]

if __name__ == "__main__":
    input_text = """
    LangChain is a powerful framework for building AI applications.
    It simplifies the creation of agents, chains, and RAG pipelines.
    """
    current = input_text
    for name, agent in agents:
        current = agent.invoke({"text": current})
        print(f"\n[{name}]\n{current}")
