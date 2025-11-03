"""
02_llm_providers.py - Working with LLM Providers

This tutorial demonstrates:
- Azure OpenAI configuration options
- Environment variable setup
- Different model parameters (temperature, max_tokens, etc.)
- Comparing different configurations

Prerequisites:
- Azure OpenAI account with deployment
- Environment variables configured
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv()

print("=== Basic Azure OpenAI Configuration ===")
# Basic configuration (uses environment variables automatically)
llm_basic = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)
response = llm_basic.invoke("Say hello in a friendly way!")
print(f"Basic config: {response.content}\n")

print("=== Advanced Azure OpenAI Configuration ===")
# Advanced configuration with explicit parameters
llm_advanced = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,  # Controls randomness: 0 = deterministic, 1 = creative
    max_tokens=150,   # Maximum tokens in response
    top_p=1.0,        # Nucleus sampling parameter
)

response = llm_advanced.invoke("Write a creative story about a robot learning to paint.")
print(f"Advanced config (creative): {response.content}\n")

print("=== Low Temperature (Deterministic) Configuration ===")
# Low temperature for more deterministic, factual responses
llm_deterministic = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.1,  # Very low temperature = more focused/consistent
    max_tokens=100,
)

response = llm_deterministic.invoke("What is 2 + 2? Explain briefly.")
print(f"Deterministic config: {response.content}\n")

print("=== High Temperature (Creative) Configuration ===")
# High temperature for creative responses
llm_creative = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.9,  # High temperature = more creative/random
    max_tokens=200,
)

response = llm_creative.invoke("Write a haiku about programming.")
print(f"Creative config: {response.content}\n")

print("=== Custom API Version Configuration ===")
# Explicitly set API version (if needed)
llm_custom = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    temperature=0.5,
)

response = llm_custom.invoke("Explain what temperature parameter does in LLM configuration.")
print(f"Custom API version: {response.content}\n")

print("Tutorial complete! You've learned about Azure OpenAI configuration options.")

