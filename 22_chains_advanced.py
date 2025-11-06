"""
22_chains_advanced.py - Advanced Chain Architectures

This tutorial demonstrates:
- Complex chain architectures
- Parallel processing
- Error recovery
- Chain composition patterns

Prerequisites:
- Understanding of basic chains and sequential chains
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain, TransformChain
from langchain.chains.base import Chain
from typing import Dict, List

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
)

print("=== Advanced Chain Patterns ===")
print("""
Advanced patterns:
1. Parallel chains
2. Conditional chains
3. Error recovery
4. Chain composition
5. Custom chain types
""")

print("=== Parallel Chain Execution ===")
# Execute multiple chains in parallel (conceptually)
# Note: True parallel execution requires async

# Create multiple independent chains
summary_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_messages([
        ("human", "Summarize this in one sentence: {text}"),
    ]),
    output_key="summary",
)

sentiment_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_messages([
        ("human", "Analyze the sentiment of this text: {text}"),
    ]),
    output_key="sentiment",
)

keywords_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_messages([
        ("human", "Extract 3 key words from: {text}"),
    ]),
    output_key="keywords",
)

# Sequential chain that processes text through all three
analysis_chain = SequentialChain(
    chains=[summary_chain, sentiment_chain, keywords_chain],
    input_variables=["text"],
    output_variables=["summary", "sentiment", "keywords"],
    verbose=True,
)

text = "LangChain is an amazing framework that makes building LLM applications easy and powerful."
result = analysis_chain({"text": text})

print(f"\nInput: {text}")
print(f"Summary: {result['summary']}")
print(f"Sentiment: {result['sentiment']}")
print(f"Keywords: {result['keywords']}\n")

print("=== Transform Chain (Pre-processing) ===")
# TransformChain processes data without LLM
def preprocess_text(inputs: Dict[str, str]) -> Dict[str, str]:
    """Preprocess text before LLM processing"""
    text = inputs["text"]
    # Simple preprocessing: clean and normalize
    processed = text.strip().lower()
    word_count = len(processed.split())
    return {
        "processed_text": processed,
        "word_count": str(word_count),
        "original_text": text,
    }

preprocess_chain = TransformChain(
    input_variables=["text"],
    output_variables=["processed_text", "word_count", "original_text"],
    transform=preprocess_text,
)

# Chain that preprocesses then analyzes
preprocess_prompt = ChatPromptTemplate.from_messages([
    ("human", "Analyze this text (word count: {word_count}): {processed_text}"),
])

analysis_llm_chain = LLMChain(
    llm=llm,
    prompt=preprocess_prompt,
    output_key="analysis",
)

preprocess_and_analyze = SequentialChain(
    chains=[preprocess_chain, analysis_llm_chain],
    input_variables=["text"],
    output_variables=["analysis"],
)

print("Transform chain example:")
result = preprocess_and_analyze({"text": "LangChain is Great For Building Apps!"})
print(f"Analysis: {result['analysis']}\n")

print("=== Conditional Chain Pattern ===")
# Pattern for conditional execution
def route_based_on_length(inputs: Dict[str, str]) -> str:
    """Route to different chains based on input length"""
    text = inputs.get("text", "")
    if len(text) > 100:
        return "detailed"  # Route to detailed analysis
    else:
        return "brief"  # Route to brief analysis

# Create different chains for different routes
brief_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_messages([
        ("human", "Give a brief analysis: {text}"),
    ]),
    output_key="analysis",
)

detailed_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_messages([
        ("human", "Give a detailed, comprehensive analysis: {text}"),
    ]),
    output_key="analysis",
)

print("Conditional routing (conceptual):")
short_text = "Python is good."
long_text = "Python is a high-level programming language known for its simplicity and readability. " * 3

print(f"Short text -> brief analysis:")
result = brief_chain.run(text=short_text)
print(f"  {result}\n")

print(f"Long text -> detailed analysis:")
result = detailed_chain.run(text=long_text[:200])
print(f"  {result[:150]}...\n")

print("=== Error Recovery Pattern ===")
# Chain with error handling
def safe_chain_execution(chain, inputs):
    """Execute chain with error recovery"""
    try:
        return chain.run(**inputs)
    except Exception as e:
        print(f"Error in chain: {e}")
        return f"Error occurred: {str(e)}"

# Example with potential error
error_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_messages([
        ("human", "Process this: {input}"),
    ]),
)

print("Error recovery example:")
result = safe_chain_execution(error_chain, {"input": "test input"})
print(f"Result: {result}\n")

print("=== Chain Composition Pattern ===")
# Compose chains into reusable components

# Component 1: Text processing
text_process = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_messages([
        ("human", "Clean and format: {raw_text}"),
    ]),
    output_key="processed_text",
)

# Component 2: Analysis
analyze = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_messages([
        ("human", "Analyze: {processed_text}"),
    ]),
    output_key="analysis",
)

# Component 3: Summary
summarize = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_messages([
        ("human", "Summarize this analysis: {analysis}"),
    ]),
    output_key="summary",
)

# Compose into pipeline
pipeline = SequentialChain(
    chains=[text_process, analyze, summarize],
    input_variables=["raw_text"],
    output_variables=["summary"],
    verbose=True,
)

print("Composed pipeline example:")
result = pipeline({"raw_text": "LangChain helps developers build AI applications efficiently."})
print(f"\nFinal summary: {result['summary']}\n")

print("=== Advanced Chain Patterns Summary ===")
print("""
1. Parallel Processing:
   - Execute independent chains
   - Combine results
   - Use async for true parallelism

2. Transform Chains:
   - Pre/post processing
   - Data transformation
   - No LLM calls

3. Conditional Routing:
   - Route based on conditions
   - Different chains for different cases
   - Dynamic execution paths

4. Error Recovery:
   - Try-except around chains
   - Fallback chains
   - Graceful degradation

5. Composition:
   - Reusable components
   - Chain libraries
   - Modular design
""")

print("=== Best Practices ===")
print("""
1. Design for reusability
   - Create focused, single-purpose chains
   - Compose them into larger workflows

2. Error handling
   - Add error recovery at each level
   - Provide meaningful error messages

3. Performance
   - Consider parallel execution where possible
   - Cache intermediate results when appropriate

4. Testing
   - Test individual chains
   - Test composed chains
   - Test error cases

5. Documentation
   - Document chain inputs/outputs
   - Explain chain purpose
   - Document dependencies
""")

print("Tutorial complete! You've learned advanced chain patterns and architectures.")

