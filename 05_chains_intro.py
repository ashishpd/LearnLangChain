"""
05_chains_intro.py - Introduction to Chains

This tutorial demonstrates:
- What chains are in LangChain
- Basic LLMChain usage
- Chaining multiple operations
- Simple sequential workflows

Prerequisites:
- Azure OpenAI setup
- Understanding of prompts and LLMs
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
)

print("=== What is a Chain? ===")
print("A chain is a sequence of operations that can be executed together.\n")

print("=== Basic Chain (LCEL) ===")
# Modern LangChain uses LCEL (LangChain Expression Language)
# Chain is created by piping prompt | llm | parser
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Explain {topic} in simple terms."),
])

# Create a chain using LCEL syntax (prompt | llm | parser)
chain = prompt | llm | StrOutputParser()

# Invoke the chain with input variables
result = chain.invoke({"topic": "quantum computing"})
print(f"Topic: quantum computing")
print(f"Explanation: {result}\n")

print("=== Chain with Multiple Variables ===")
# Chains can handle multiple input variables
multi_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code review assistant."),
    ("human", "Review this {language} code for {aspect}:\n\n{code}"),
])

review_chain = multi_prompt | llm | StrOutputParser()

code_sample = """
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""

result = review_chain.invoke({
    "language": "Python",
    "aspect": "performance and readability",
    "code": code_sample
})
print(f"Code Review Result:\n{result}\n")

print("=== Using invoke() Method ===")
# invoke() is the standard method for running chains
# It returns the parsed output directly (string when using StrOutputParser)
prompt = ChatPromptTemplate.from_messages([
    ("human", "Write a {length} {type} story about {topic}."),
])

story_chain = prompt | llm | StrOutputParser()

# Using invoke() returns the parsed string output
result = story_chain.invoke({
    "length": "short",
    "type": "science fiction",
    "topic": "time travel"
})
print(f"Story result: {result}\n")

print("=== Chain with Different Temperatures ===")
# You can create chains with different LLM configurations
creative_llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.9,  # More creative
)

factual_llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.1,  # More factual
)

# Creative chain for stories
creative_prompt = ChatPromptTemplate.from_messages([
    ("human", "Write a creative story about {topic}."),
])
creative_chain = creative_prompt | creative_llm | StrOutputParser()

# Factual chain for explanations
factual_prompt = ChatPromptTemplate.from_messages([
    ("human", "Explain {topic} accurately and factually."),
])
factual_chain = factual_prompt | factual_llm | StrOutputParser()

print("Creative response:")
creative_result = creative_chain.invoke({"topic": "space exploration"})
print(f"{creative_result}\n")

print("Factual response:")
factual_result = factual_chain.invoke({"topic": "space exploration"})
print(f"{factual_result}\n")

print("=== Reusable Chains ===")
# Chains can be saved and reused with different inputs
greeting_prompt = ChatPromptTemplate.from_messages([
    ("human", "Write a {tone} greeting for {occasion}."),
])

greeting_chain = greeting_prompt | llm | StrOutputParser()

# Reuse the same chain with different inputs
greetings = [
    {"tone": "formal", "occasion": "a business meeting"},
    {"tone": "casual", "occasion": "a friend's birthday"},
    {"tone": "professional", "occasion": "a conference presentation"},
]

print("Multiple greetings using the same chain:")
for greeting_input in greetings:
    result = greeting_chain.invoke(greeting_input)
    print(f"  {greeting_input['tone']} for {greeting_input['occasion']}: {result}")

print("\nTutorial complete! You've learned the basics of LangChain chains.")

