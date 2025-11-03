"""
10_agents_intro.py - Introduction to Agents

This tutorial demonstrates:
- What agents are in LangChain
- Different agent types (zero-shot-react, react-docstore, etc.)
- Basic agent setup and usage
- Agent execution flow

Prerequisites:
- Understanding of chains, prompts, and memory
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.agents import load_tools

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0,
)

print("=== What is an Agent? ===")
print("""
An agent is an LLM that can:
- Use tools to interact with the world
- Make decisions about which tools to use
- Chain multiple tool calls together
- Reason about the results

Unlike simple chains, agents can decide what actions to take based on the task.
""")

print("=== Basic Agent Setup ===")
# Load some built-in tools
# Note: Some tools require API keys (like SERPAPI for web search)
# For this example, we'll use simpler tools that don't require extra setup

try:
    # Try loading tools (may require API keys)
    tools = load_tools(["llm-math"], llm=llm)
    
    # Initialize a zero-shot-react agent
    # ZERO_SHOT_REACT_DESCRIPTION: Uses ReAct framework, no memory
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,  # Set to True to see agent's reasoning
    )
    
    print("\n=== Example 1: Math Calculation ===")
    # Agent will use the math tool
    result = agent.run("What is 15 multiplied by 8, then divided by 3?")
    print(f"Result: {result}\n")
    
except Exception as e:
    print(f"Note: Tool loading requires additional setup. Error: {e}\n")
    print("For this tutorial, we'll demonstrate agent concepts without external tools.\n")

print("=== Agent Types Explained ===")
print("""
1. ZERO_SHOT_REACT_DESCRIPTION:
   - No memory, uses ReAct framework
   - Good for: Single-turn tasks, tool usage

2. REACT_DOCSTORE:
   - For querying document stores
   - Good for: Question answering over documents

3. CONVERSATIONAL_REACT_DESCRIPTION:
   - Has memory, maintains conversation
   - Good for: Multi-turn conversations with tools

4. CHAT_ZERO_SHOT_REACT_DESCRIPTION:
   - Chat-based agent (uses ChatModel)
   - Good for: Conversational interfaces

5. CHAT_CONVERSATIONAL_REACT_DESCRIPTION:
   - Chat-based with memory
   - Good for: Long conversations with tool usage
""")

print("=== Agent Execution Flow ===")
print("""
When an agent runs:
1. Agent receives a question/task
2. Agent decides which tool to use (or if it can answer directly)
3. Agent calls the tool with appropriate parameters
4. Agent receives tool output
5. Agent reasons about the result
6. Agent either:
   - Returns final answer, or
   - Uses another tool if more information needed
""")

print("=== Simple Agent Simulation ===")
# Create a conceptual example showing agent reasoning
# In practice, you'd use initialize_agent with real tools

print("""
Example Agent Reasoning Flow:

Task: "What is the capital of France, and what is 5 * 5?"

Step 1: Agent thinks: "I need to answer two questions:
      1. Capital of France - I know this is Paris
      2. 5 * 5 - I need to use a calculator tool"

Step 2: Agent answers first question: "The capital of France is Paris."

Step 3: Agent uses calculator tool: calculate(5, *, 5) = 25

Step 4: Agent combines answers: "The capital of France is Paris, and 5 * 5 = 25"
""")

print("=== Agent vs Chain Comparison ===")
print("""
Chains:
- Fixed execution path
- Predictable flow
- Good for: Structured workflows

Agents:
- Dynamic execution path
- Can choose tools adaptively
- Good for: Tasks requiring reasoning and tool selection
""")

print("=== Next Steps ===")
print("""
To use agents effectively:
1. Define or load appropriate tools (next tutorial)
2. Choose the right agent type for your use case
3. Provide clear instructions to the agent
4. Use verbose=True to debug agent reasoning

In the next tutorial, we'll learn how to create custom tools.
""")

print("Tutorial complete! You've learned the basics of LangChain agents.")

