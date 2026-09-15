"""
02_agent_with_tools.py
-----------------------
Topic: Agent with Multiple Tools using LangChain (Gemini)
What:    One agent bound to multiple tools; picks the right one per query
Meaning: LLM sees both tools and decides which to call based on the question
Example: "15 + 27?" → add_numbers | "Capital of India?" → get_capital
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# ── Tools ─────────────────────────────────────────────────────────────────────
@tool
def add_numbers(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b

@tool
def get_capital(country: str) -> str:
    """Returns the capital of a country."""
    capitals = {"india": "New Delhi", "france": "Paris", "usa": "Washington D.C."}
    return capitals.get(country.lower(), "Unknown")

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

# ── Prompt ────────────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# ── Agent ─────────────────────────────────────────────────────────────────────
tools = [add_numbers, get_capital]
agent = create_tool_calling_agent(llm, tools, prompt)   # works with Gemini
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    print(executor.invoke({"input": "What is 15 + 27?"})["output"])
    print(executor.invoke({"input": "What is the capital of India?"})["output"])
