"""
09_memory_types.py - Different Memory Types

This tutorial demonstrates:
- Different types of memory in LangChain
- ConversationBufferMemory (full history)
- ConversationSummaryMemory (summarized history)
- ConversationBufferWindowMemory (sliding window)
- ConversationTokenBufferMemory (token-limited)
- When to use each type

Prerequisites:
- Understanding of basic memory
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_classic.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
)
from langchain_classic.chains import ConversationChain
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
)

template = """The following is a friendly conversation between a human and an AI.
The AI is helpful and provides detailed answers.

{history}
Human: {input}
AI:"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)

print("=== 1. ConversationBufferMemory ===")
print("Stores the entire conversation history without limits.")
print("Best for: Short conversations, when full context is needed\n")

buffer_memory = ConversationBufferMemory()
buffer_chain = ConversationChain(
    llm=llm,
    memory=buffer_memory,
    prompt=prompt,
    verbose=False
)

# Simulate conversation
for i in range(3):
    response = buffer_chain.predict(input=f"Tell me fact number {i+1} about Python")
    print(f"Fact {i+1}: {response[:100]}...\n")

print(f"Buffer memory size: {len(buffer_memory.buffer)} characters\n")

print("=== 2. ConversationBufferWindowMemory ===")
print("Stores only the last N interactions (sliding window).")
print("Best for: Long conversations where recent context matters most\n")

# Keep only last 2 exchanges
window_memory = ConversationBufferWindowMemory(k=2)
window_chain = ConversationChain(
    llm=llm,
    memory=window_memory,
    prompt=prompt,
    verbose=False
)

# Simulate longer conversation
for i in range(4):
    window_chain.predict(input=f"Topic {i+1}: Tell me about machine learning")

print("Window memory (last 2 exchanges):")
print(window_memory.buffer)
print()

print("=== 3. ConversationTokenBufferMemory ===")
print("Stores conversation up to a token limit.")
print("Best for: Controlling memory size by tokens rather than messages\n")

# Limit to approximately 100 tokens
token_memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=100,
    return_messages=True
)

token_chain = ConversationChain(
    llm=llm,
    memory=token_memory,
    prompt=prompt,
    verbose=False
)

for i in range(3):
    token_chain.predict(input=f"Explain concept {i+1} in detail")

print(f"Token buffer memory length: {len(token_memory.buffer)} messages\n")

print("=== 4. ConversationSummaryMemory ===")
print("Summarizes old messages, keeps recent ones in full.")
print("Best for: Very long conversations where you need context but want to save tokens\n")

summary_memory = ConversationSummaryMemory(llm=llm)
summary_chain = ConversationChain(
    llm=llm,
    memory=summary_memory,
    prompt=prompt,
    verbose=False
)

# Build up conversation history
topics = [
    "Python programming",
    "Web development",
    "Data science",
    "Machine learning",
    "Cloud computing"
]

print("Building conversation with summary memory:")
for topic in topics:
    response = summary_chain.predict(input=f"Tell me about {topic}")
    print(f"Discussed: {topic}")

print("\nSummary of conversation:")
print(summary_memory.moving_summary_buffer)
print()

print("=== Memory Comparison Example ===")
# Compare different memory types with the same conversation

conversations = [
    ("Buffer", ConversationBufferMemory()),
    ("Window (k=1)", ConversationBufferWindowMemory(k=1)),
    ("Summary", ConversationSummaryMemory(llm=llm)),
]

test_inputs = [
    "My favorite color is blue",
    "What programming language do you like?",
    "I'm a software engineer",
    "What's my favorite color?"
]

for name, memory in conversations:
    print(f"\n{name} Memory:")
    chain = ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=False)
    
    for inp in test_inputs:
        if "favorite color" in inp.lower():
            response = chain.predict(input=inp)
            print(f"  Q: {inp}")
            print(f"  A: {response[:80]}...")

print("\n=== When to Use Each Memory Type ===")
print("""
1. BufferMemory: Short conversations, chatbots, when you need full history
2. WindowMemory: Long conversations, when only recent context matters
3. TokenMemory: When you want precise control over memory size in tokens
4. SummaryMemory: Very long conversations, cost optimization, document Q&A
""")

print("Tutorial complete! You've learned about different memory types and when to use them.")

