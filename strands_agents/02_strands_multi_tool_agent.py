"""
Strands SDK – Agent with Multiple Tools
----------------------------------------
Key concepts:
  - Multiple @tool functions passed to Agent
  - Tools can call external logic (simulated here)
  - Agent picks the right tool based on the prompt
"""

from strands import Agent, tool


# ── Tools ─────────────────────────────────────────────────────────────────────
@tool
def get_weather(city: str) -> str:
    """Return a mock weather report for a city."""
    return f"The weather in {city} is sunny and 25°C."


@tool
def get_population(city: str) -> str:
    """Return a mock population for a city."""
    populations = {"Delhi": "32M", "Mumbai": "21M", "Bangalore": "13M"}
    return f"{city} has a population of {populations.get(city, 'unknown')}."


# ── Agent ─────────────────────────────────────────────────────────────────────
agent = Agent(tools=[get_weather, get_population])

if __name__ == "__main__":
    print(agent("What is the weather in Delhi?"))
    print(agent("How many people live in Mumbai?"))
