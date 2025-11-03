"""
12_multi_tool_agents.py - Agents with Multiple Tools

This tutorial demonstrates:
- Agents using multiple tools in sequence
- Tool selection and chaining
- Complex agent workflows
- Error handling in multi-tool scenarios

Prerequisites:
- Understanding of agents and custom tools
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0,
)

print("=== Multi-Tool Agent Setup ===")
# Create a set of diverse tools for the agent

# Tool 1: Text processing
def count_words(text: str) -> str:
    """Counts the number of words in text."""
    words = text.split()
    return f"The text has {len(words)} words."

# Tool 2: Text transformation
def uppercase_text(text: str) -> str:
    """Converts text to uppercase."""
    return text.upper()

# Tool 3: Information lookup
def get_info(topic: str) -> str:
    """Gets information about a topic."""
    info_db = {
        "python": "Python is a high-level programming language known for simplicity.",
        "langchain": "LangChain is a framework for building LLM applications.",
        "ai": "Artificial Intelligence enables machines to learn and make decisions.",
    }
    return info_db.get(topic.lower(), f"Information about {topic} not found in database.")

# Tool 4: Calculator
def calculate(expression: str) -> str:
    """Evaluates a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Error: Invalid expression"

# Tool 5: Text analysis
def analyze_sentiment(text: str) -> str:
    """Analyzes sentiment of text (simplified version)."""
    positive_words = ["good", "great", "excellent", "amazing", "happy"]
    negative_words = ["bad", "terrible", "awful", "sad", "disappointed"]
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return "Positive sentiment"
    elif neg_count > pos_count:
        return "Negative sentiment"
    else:
        return "Neutral sentiment"

# Create Tool objects
tools = [
    Tool(
        name="count_words",
        func=count_words,
        description="Useful for counting the number of words in a text. Input: a string of text."
    ),
    Tool(
        name="uppercase_text",
        func=uppercase_text,
        description="Useful for converting text to uppercase. Input: a string of text."
    ),
    Tool(
        name="get_info",
        func=get_info,
        description="Useful for looking up information about topics like Python, LangChain, or AI. Input: a topic name."
    ),
    Tool(
        name="calculator",
        func=calculate,
        description="Useful for performing calculations. Input: a mathematical expression like '2+2'."
    ),
    Tool(
        name="analyze_sentiment",
        func=analyze_sentiment,
        description="Useful for analyzing the sentiment of text. Input: a string of text."
    ),
]

print("Available tools:")
for i, tool in enumerate(tools, 1):
    print(f"  {i}. {tool.name}: {tool.description}")

# Initialize agent with all tools
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=10,  # Limit iterations to prevent infinite loops
)

print("\n=== Example 1: Sequential Tool Usage ===")
# Agent will use multiple tools in sequence
result = agent.run(
    "Get information about Python, count the words in that information, "
    "and then calculate what 10 * 5 equals"
)
print(f"\nFinal Answer: {result}\n")

print("=== Example 2: Complex Multi-Step Task ===")
result = agent.run(
    "Analyze the sentiment of 'I feel great about this amazing product', "
    "then convert that analysis to uppercase, and count how many words it has"
)
print(f"\nFinal Answer: {result}\n")

print("=== Example 3: Conditional Tool Selection ===")
# Agent needs to decide which tools to use based on the question
result = agent.run(
    "What information do you have about LangChain? "
    "After getting that, calculate 15 + 25"
)
print(f"\nFinal Answer: {result}\n")

print("=== Example 4: Error Handling ===")
# Agent should handle errors gracefully
result = agent.run(
    "Calculate 'hello + world' (this will error), then calculate 5 * 5 correctly"
)
print(f"\nFinal Answer: {result}\n")

print("=== Agent Reasoning with Multiple Tools ===")
print("""
When an agent has multiple tools:
1. Agent analyzes the query
2. Agent identifies which tools are needed
3. Agent determines the order of tool execution
4. Agent chains tool outputs if needed
5. Agent synthesizes final answer

The agent's reasoning is visible when verbose=True
""")

print("=== Best Practices for Multi-Tool Agents ===")
print("""
1. Clear tool descriptions: Help agent choose the right tool
2. Tool naming: Use descriptive, distinct names
3. Error handling: Tools should return error messages, not exceptions
4. Limit iterations: Use max_iterations to prevent infinite loops
5. Tool specificity: Make tools focused rather than general-purpose
6. Test individually: Ensure each tool works before combining
""")

print("\n=== Advanced: Tool with Dependency ===")
# Some tools might depend on outputs from other tools

def format_result(text: str, format_type: str) -> str:
    """Formats text in different ways."""
    if format_type.lower() == "uppercase":
        return text.upper()
    elif format_type.lower() == "lowercase":
        return text.lower()
    elif format_type.lower() == "title":
        return text.title()
    else:
        return text

format_tool = Tool(
    name="format_result",
    func=lambda x: format_result(x.split("|")[0], x.split("|")[1]) if "|" in x else format_result(x, "uppercase"),
    description="Formats text. Input format: 'text|format_type' where format_type is uppercase, lowercase, or title."
)

advanced_tools = [get_info, calculator, format_tool]
advanced_tool_objects = [
    Tool(name="get_info", func=get_info, description="Get information about topics."),
    Tool(name="calculator", func=calculate, description="Perform calculations."),
    format_tool,
]

advanced_agent = initialize_agent(
    tools=advanced_tool_objects,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

result = advanced_agent.run(
    "Get info about Python, then format it in uppercase"
)
print(f"\nFinal Answer: {result}\n")

print("Tutorial complete! You've learned how to build agents with multiple tools.")

