"""
13_agent_memory.py - Agents with Memory

This tutorial demonstrates:
- Combining agents with memory
- Conversational agents that remember context
- Stateful agent interactions
- Memory management in agents

Prerequisites:
- Understanding of agents, tools, and memory
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ChatMessageHistory

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
)

print("=== Why Agents Need Memory ===")
print("""
Agents with memory can:
- Remember previous conversation context
- Maintain state across interactions
- Build on previous tool results
- Provide personalized responses
""")

print("=== Simple Tools for Demo ===")
# Create some simple tools
def get_user_info(user_id: str) -> str:
    """Gets information about a user."""
    users = {
        "alice": "Alice is a software engineer who loves Python.",
        "bob": "Bob is a data scientist interested in machine learning.",
    }
    return users.get(user_id.lower(), f"User {user_id} not found.")

def calculate(expression: str) -> str:
    """Performs calculations."""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Error: Invalid expression"

def remember_fact(topic: str, fact: str) -> str:
    """Stores a fact about a topic."""
    return f"Remembered: {topic} - {fact}"

tools = [
    Tool(
        name="get_user_info",
        func=get_user_info,
        description="Gets information about a user. Input: user ID like 'alice' or 'bob'."
    ),
    Tool(
        name="calculator",
        func=calculate,
        description="Performs calculations. Input: mathematical expression."
    ),
    Tool(
        name="remember_fact",
        func=remember_fact,
        description="Stores a fact. Input format: 'topic|fact' where topic and fact are separated by |."
    ),
]

print("=== Agent with ConversationBufferMemory ===")
# Create memory for the agent
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create conversational agent with memory
# CONVERSATIONAL_REACT_DESCRIPTION includes memory support
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
)

print("\n=== Conversation Example ===")
print("Starting a conversation with the agent...\n")

# First message
response = agent.run(input="Hi, my name is Alice. Can you tell me about myself?")
print(f"Agent: {response}\n")

# Second message - agent should remember the name
response = agent.run(input="What's my name?")
print(f"Agent: {response}\n")

# Third message - using calculator
response = agent.run(input="Can you calculate 15 * 8 for me?")
print(f"Agent: {response}\n")

# Fourth message - agent should remember previous calculation context
response = agent.run(input="What calculation did we just do?")
print(f"Agent: {response}\n")

print("=== Memory Contents ===")
# Check what's stored in memory
print("Current memory buffer:")
print(memory.buffer)
print()

print("=== Resetting Conversation ===")
# Clear memory and start fresh
memory.clear()

print("Memory cleared. Starting new conversation...\n")
response = agent.run(input="Hello! Can you help me calculate 10 + 5?")
print(f"Agent: {response}\n")

print("=== Multi-Turn Tool Usage ===")
# Agent using tools across multiple turns while maintaining context
memory.clear()

conversation_turns = [
    "I'd like to know about user 'bob'",
    "What did we learn about bob?",
    "Now calculate 20 * 3",
    "Summarize what we've done so far in this conversation",
]

for i, turn in enumerate(conversation_turns, 1):
    print(f"Turn {i}: {turn}")
    response = agent.run(input=turn)
    print(f"Agent: {response}\n")

print("=== Memory Management ===")
print("""
Memory in agents:
- Stores conversation history
- Includes both user inputs and agent outputs
- Can include tool execution results
- Persists across agent.run() calls
- Can be cleared with memory.clear()
""")

print("=== Best Practices ===")
print("""
1. Use appropriate agent type:
   - CONVERSATIONAL_REACT_DESCRIPTION: For chat-based agents with tools
   - CHAT_CONVERSATIONAL_REACT_DESCRIPTION: For ChatModels with memory

2. Memory management:
   - Clear memory when starting new conversations
   - Consider memory limits for long conversations
   - Use appropriate memory types (Buffer, Summary, etc.)

3. Tool design:
   - Design tools that work well with conversational context
   - Tool outputs should be clear for the agent

4. Context window:
   - Long conversations can exceed context limits
   - Consider using summary memory for very long conversations
""")

print("=== Advanced: Custom Memory Integration ===")
# You can customize how memory interacts with agents

class CustomMemory:
    """Simple custom memory example"""
    def __init__(self):
        self.messages = []
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    
    def get_history(self) -> str:
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.messages])

# Note: In practice, use LangChain's built-in memory classes
# This is just for demonstration

print("\nCustom memory pattern (conceptual):")
custom_mem = CustomMemory()
custom_mem.add_message("user", "Hello")
custom_mem.add_message("assistant", "Hi! How can I help?")
print(custom_mem.get_history())

print("\nTutorial complete! You've learned how to combine agents with memory.")

