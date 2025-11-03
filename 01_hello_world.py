"""
01_hello_world.py - Introduction to LangChain

This is the first tutorial in the LangChain series. It demonstrates:
- How to set up Azure OpenAI with LangChain
- Basic LLM invocation
- Simple text generation

Prerequisites:
- Azure OpenAI account and deployment
- Set environment variables (see .env.example)
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize Azure OpenAI LLM
# This is the basic setup for connecting to Azure OpenAI
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)

# Simple example: Generate text
print("=== Hello World Example ===")
response = llm.invoke("Hello! Can you tell me a fun fact about Python programming?")
print(f"Response: {response.content}\n")

# Another example: Simple question answering
print("=== Question Answering Example ===")
question = "What is LangChain?"
response = llm.invoke(question)
print(f"Question: {question}")
print(f"Answer: {response.content}\n")

# Example with a more detailed prompt
print("=== Detailed Prompt Example ===")
prompt = """
Explain the concept of machine learning in simple terms.
Use analogies that a beginner can understand.
Keep it to 2-3 sentences.
"""
response = llm.invoke(prompt)
print(f"Response: {response.content}\n")

print("Congratulations! You've completed your first LangChain tutorial!")

