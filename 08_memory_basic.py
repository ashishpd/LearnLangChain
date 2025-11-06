"""
08_memory_basic.py - Basic Memory

This tutorial demonstrates:
- Introduction to Memory in LangChain
- ConversationBufferMemory
- Basic chat examples with memory
- Maintaining conversation context

Prerequisites:
- Understanding of chains and prompts
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
)

print("=== What is Memory? ===")
print("Memory allows LangChain chains to remember information from previous interactions.\n")

print("=== Conversation Buffer Memory ===")
# ConversationBufferMemory stores the entire conversation history

# Create memory instance
memory = ConversationBufferMemory()

# Basic memory operations
memory.save_context(
    {"input": "Hi, my name is Alice"},
    {"output": "Hello Alice! Nice to meet you. How can I help you today?"}
)
memory.save_context(
    {"input": "What's the weather like?"},
    {"output": "I don't have access to real-time weather data, but I'd be happy to help you with other questions!"}
)

# Retrieve conversation history
print("Conversation history:")
print(memory.buffer)
print()

# Clear memory
memory.clear()

print("=== Conversation Chain with Memory ===")
# ConversationChain automatically handles memory in conversations

# Create a template that includes memory
template = """The following is a friendly conversation between a human and an AI. 
The AI is talkative and provides lots of specific details from its context.
If the AI doesn't know the answer to a question, it truthfully says it doesn't know.

Current conversation:
{history}
Human: {input}
AI:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

# Create conversation chain with memory
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    prompt=prompt,
    verbose=True
)

print("Starting conversation with memory:\n")

# Simulate a conversation
responses = [
    "Hi! I'm learning about LangChain. Can you tell me what it is?",
    "That's interesting! Can you give me an example of how to use it?",
    "What about memory? How does that work?",
]

for user_input in responses:
    print(f"Human: {user_input}")
    response = conversation.predict(input=user_input)
    print(f"AI: {response}\n")

print("=== Memory Persistence Check ===")
# Verify that memory persists across multiple calls
print("Asking about previous context:")
result = conversation.predict(input="Can you remind me what we talked about earlier?")
print(f"AI: {result}\n")

print("=== Multiple Conversation Instances ===")
# Each conversation chain maintains its own memory
alice_conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    verbose=False
)

bob_conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    verbose=False
)

print("Alice's conversation:")
alice_conversation.predict(input="Hi, my name is Alice")
response = alice_conversation.predict(input="What's my name?")
print(f"Response: {response}\n")

print("Bob's conversation:")
bob_conversation.predict(input="Hi, my name is Bob")
response = bob_conversation.predict(input="What's my name?")
print(f"Response: {response}\n")

print("=== Manual Memory Management ===")
# You can manually manage memory contents
custom_memory = ConversationBufferMemory(return_messages=True)

# Add messages manually
custom_memory.chat_memory.add_user_message("I love Python programming")
custom_memory.chat_memory.add_ai_message("That's great! Python is a versatile language.")

# Get memory as dictionary
memory_dict = custom_memory.load_memory_variables({})
print("Manual memory content:")
print(memory_dict)
print()

print("Tutorial complete! You've learned the basics of memory in LangChain.")

