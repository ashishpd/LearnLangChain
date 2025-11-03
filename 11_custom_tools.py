"""
11_custom_tools.py - Creating Custom Tools

This tutorial demonstrates:
- Creating custom tools for agents
- Tool definitions and descriptions
- Tool execution
- Integrating tools with agents

Prerequisites:
- Understanding of agents
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.agents import AgentExecutor
from langchain.prompts import PromptTemplate
from typing import Optional

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0,
)

print("=== What are Tools? ===")
print("Tools are functions that agents can use to interact with the world.")
print("Tools extend an agent's capabilities beyond what the LLM knows.\n")

print("=== Creating Simple Tools ===")

# Example 1: Simple function tool
def get_word_length(word: str) -> str:
    """Returns the length of a word.
    
    Args:
        word: The word to measure
        
    Returns:
        The length of the word
    """
    return str(len(word))

# Example 2: Calculator tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression.
    
    Args:
        expression: A mathematical expression (e.g., "2 + 2", "10 * 5")
        
    Returns:
        The result of the calculation
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Example 3: Text manipulation tool
def reverse_text(text: str) -> str:
    """Reverses a given string.
    
    Args:
        text: The text to reverse
        
    Returns:
        The reversed text
    """
    return text[::-1]

# Example 4: Weather simulation tool (mock)
def get_weather(city: str) -> str:
    """Gets the current weather for a city.
    
    Args:
        city: The name of the city
        
    Returns:
        Weather information for the city
    """
    # Mock weather data
    weather_data = {
        "paris": "Sunny, 20°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 25°C",
        "new york": "Sunny, 18°C",
    }
    city_lower = city.lower()
    return weather_data.get(city_lower, f"Weather data not available for {city}")

print("=== Converting Functions to Tools ===")
# Convert functions to LangChain Tool objects

word_length_tool = Tool(
    name="word_length",
    func=get_word_length,
    description="Useful when you need to find the length of a word. Input should be a single word."
)

calculator_tool = Tool(
    name="calculator",
    func=calculator,
    description="Useful for performing mathematical calculations. Input should be a mathematical expression like '2+2' or '10*5'."
)

reverse_tool = Tool(
    name="reverse_text",
    func=reverse_text,
    description="Useful when you need to reverse a string. Input should be the text to reverse."
)

weather_tool = Tool(
    name="get_weather",
    func=get_weather,
    description="Useful when you need to find weather information for a city. Input should be the city name."
)

# List of tools
tools = [word_length_tool, calculator_tool, reverse_tool, weather_tool]

print("Tools created:")
for tool in tools:
    print(f"  - {tool.name}: {tool.description}")

print("\n=== Using Tools with Agent ===")
# Initialize agent with custom tools
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

print("\n=== Example 1: Using Calculator Tool ===")
result = agent.run("What is 25 multiplied by 4?")
print(f"Final Answer: {result}\n")

print("=== Example 2: Using Multiple Tools ===")
result = agent.run("What is the length of the word 'hello' reversed?")
print(f"Final Answer: {result}\n")

print("=== Example 3: Using Weather Tool ===")
result = agent.run("What's the weather like in Paris?")
print(f"Final Answer: {result}\n")

print("=== Creating Advanced Tools ===")
# Tools with multiple parameters require structured inputs

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    """Input for search tool"""
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Maximum number of results")

def search_function(query: str, max_results: int = 5) -> str:
    """Searches for information.
    
    Args:
        query: The search query
        max_results: Maximum number of results
        
    Returns:
        Search results
    """
    # Mock search results
    results = [
        f"Result {i+1}: Information about '{query}' - Detail {i+1}"
        for i in range(min(max_results, 3))
    ]
    return "\n".join(results)

# Structured tool with Pydantic model
search_tool = StructuredTool.from_function(
    func=search_function,
    name="search",
    description="Searches for information. Use this when you need to find information about a topic.",
    args_schema=SearchInput,
)

print("\n=== Advanced Tool with Structured Input ===")
advanced_tools = [search_tool, calculator_tool]

advanced_agent = initialize_agent(
    tools=advanced_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

result = advanced_agent.run("Search for information about Python and then calculate 10 * 10")
print(f"Final Answer: {result}\n")

print("=== Tool Best Practices ===")
print("""
1. Write clear descriptions: Agents use descriptions to decide when to use tools
2. Handle errors gracefully: Tools should return error messages, not crash
3. Keep tools focused: Each tool should do one thing well
4. Use type hints: Helps with validation and documentation
5. Provide examples: Include examples in descriptions when helpful
""")

print("Tutorial complete! You've learned how to create and use custom tools with agents.")

