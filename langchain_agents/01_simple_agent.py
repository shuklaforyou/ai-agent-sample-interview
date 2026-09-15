"""
01_simple_agent.py
------------------
Topic: Single AI Agent using LangChain (Gemini)
What:    Minimal agent with one tool using native tool-calling
Meaning: LLM reasons → picks a tool → observes result → answers
Example: "Weather in Delhi?" → get_weather("Delhi") → "Sunny, 25°C"

Note: Gemini uses native function calling (not ReAct text format).
      create_tool_calling_agent is the correct choice for Gemini.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

load_dotenv()

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

# ── Tool ──────────────────────────────────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """Returns fake weather for a given city."""
    return f"The weather in {city} is sunny, 25°C."

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

# ── Prompt ────────────────────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

# ── Agent + Executor ──────────────────────────────────────────────────────────
tools = [get_weather]
agent = create_tool_calling_agent(llm, tools, prompt)   # works with Gemini
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    result = executor.invoke({"input": "What is the weather in Delhi?"})
    print(result["output"])
