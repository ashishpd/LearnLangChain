"""
26_debugging.py - Debugging LangChain Applications

This tutorial demonstrates:
- Debugging strategies
- Verbose mode
- Troubleshooting tips
- Common issues and solutions

Prerequisites:
- Basic LangChain knowledge
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains import SequentialChain
from langchain.callbacks import StdOutCallbackHandler

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
)

print("=== Debugging Strategies ===")
print("""
Effective debugging:
1. Use verbose mode
2. Check inputs/outputs at each step
3. Use callbacks for monitoring
4. Isolate components
5. Check error messages carefully
""")

print("=== Verbose Mode ===")
# Verbose mode shows execution details

prompt = ChatPromptTemplate.from_messages([
    ("human", "Explain {topic} briefly."),
])

chain = prompt | llm | StrOutputParser()

print("Running chain with verbose=True:")
# For verbose output with LCEL, use callbacks
result = chain.invoke(
    {"topic": "Python"},
    config={"callbacks": [StdOutCallbackHandler()]}
)
print(f"\nResult: {result}\n")

print("=== Step-by-Step Debugging ===")
# Break down chain execution to debug

print("Debugging approach:")
print("1. Test individual components")

# Test prompt formatting
test_input = {"topic": "AI"}
formatted = prompt.format_messages(**test_input)
print(f"\nFormatted prompt: {formatted[0].content}")

# Test LLM directly
print("\n2. Test LLM directly:")
llm_response = llm.invoke(formatted)
print(f"LLM response: {llm_response.content[:100]}...")

# Test full chain
print("\n3. Test full chain:")
chain_result = chain.invoke(test_input)
print(f"Chain result: {chain_result}\n")

print("=== Debugging with Callbacks ===")
# Use callbacks to see what's happening

class DebugCallback(StdOutCallbackHandler):
    """Enhanced callback for debugging"""
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"[DEBUG] Chain started")
        print(f"[DEBUG] Inputs: {inputs}")
    
    def on_chain_end(self, outputs, **kwargs):
        print(f"[DEBUG] Chain ended")
        print(f"[DEBUG] Outputs: {outputs}")

debug_callback = DebugCallback()

debug_chain = prompt | llm | StrOutputParser()

print("Debugging with callbacks:")
result = debug_chain.invoke(
    {"topic": "LangChain"},
    config={"callbacks": [debug_callback]}
)
print()

print("=== Common Issues and Solutions ===")
print("""
1. Authentication Errors:
   - Check environment variables
   - Verify Azure OpenAI credentials
   - Check API endpoint and deployment name

2. Rate Limiting:
   - Add delays between requests
   - Implement retry logic
   - Check API quotas

3. Unexpected Outputs:
   - Check prompt formatting
   - Verify input variables
   - Review temperature settings

4. Memory Issues:
   - Check memory configuration
   - Clear memory when needed
   - Monitor context window size

5. Chain Execution Errors:
   - Verify input/output keys
   - Check chain sequence
   - Validate data types
""")

print("=== Input/Output Validation ===")
# Validate inputs and outputs

def validate_chain_inputs(chain, inputs):
    """Validate inputs for a chain"""
    required_keys = set(chain.input_keys)
    provided_keys = set(inputs.keys())
    
    missing = required_keys - provided_keys
    extra = provided_keys - required_keys
    
    if missing:
        print(f"ERROR: Missing inputs: {missing}")
        return False
    
    if extra:
        print(f"WARNING: Extra inputs: {extra}")
    
    return True

# Test validation
test_prompt = ChatPromptTemplate.from_messages([
    ("human", "Process: {input}"),
])
test_chain = test_prompt | llm | StrOutputParser()

print("Input validation:")
# LCEL chains don't have input_keys, so we validate the prompt instead
try:
    test_prompt.format_messages(input="test")  # Validate prompt can format
    result = test_chain.invoke({"input": "test"})
    print(f"Result: {result}\n")
except Exception as e:
    print(f"Validation error: {e}\n")

# Test with wrong key
try:
    test_chain.invoke({"wrong_key": "test"})
except Exception as e:
    print(f"Expected error with wrong key: {type(e).__name__}\n")

print("=== Debugging Sequential Chains ===")
# Debug multi-step chains

chain1_prompt = ChatPromptTemplate.from_messages([
    ("human", "Summarize: {text}"),
])
chain1 = chain1_prompt | llm | StrOutputParser()

chain2_prompt = ChatPromptTemplate.from_messages([
    ("human", "Analyze this summary: {summary}"),
])
chain2 = chain2_prompt | llm | StrOutputParser()

# Note: SequentialChain requires LLMChain instances, not LCEL chains
# For LCEL, use RunnablePassthrough for sequential operations
# This is a simplified example - full LCEL sequential patterns use different syntax
from langchain_core.runnables import RunnablePassthrough
from typing import Dict

def summarize_then_analyze(inputs: Dict) -> Dict:
    """Sequential processing with LCEL"""
    summary = chain1.invoke(inputs)
    analysis = chain2.invoke({"summary": summary})
    return {"summary": summary, "analysis": analysis}

sequential = summarize_then_analyze

print("Debugging sequential chain:")
result = sequential({"text": "LangChain is a framework for building LLM applications."})
print(f"\nFinal outputs: {result}\n")

print("=== Error Handling and Debugging ===")
# Handle and debug errors

def safe_chain_execution(chain, inputs):
    """Execute chain with error handling and debugging"""
    try:
        print(f"[DEBUG] Attempting to execute chain with inputs: {inputs}")
        result = chain.invoke(inputs)
        print(f"[DEBUG] Success: {result}")
        return result
    except KeyError as e:
        print(f"[DEBUG] KeyError: Missing key {e}")
        print(f"[DEBUG] Expected keys: {chain.input_keys}")
        return None
    except Exception as e:
        print(f"[DEBUG] Error: {type(e).__name__}: {e}")
        return None

print("Safe execution with debugging:")
result = safe_chain_execution(test_chain, {"input": "test input"})
print()

print("=== Debugging Tips ===")
print("""
1. Start Simple:
   - Test with minimal examples
   - Verify each component works
   - Build up complexity gradually

2. Use Print Statements:
   - Print inputs at each step
   - Print intermediate outputs
   - Print final results

3. Check Types:
   - Verify input types
   - Check output types
   - Ensure compatibility

4. Isolate Problems:
   - Test components separately
   - Narrow down to specific issue
   - Remove unrelated code

5. Read Error Messages:
   - Understand error types
   - Check stack traces
   - Look for specific error details

6. Use Tools:
   - Logging libraries
   - Debuggers (pdb)
   - Monitoring tools
""")

print("=== Environment Debugging ===")
# Check environment setup

print("Environment check:")
print(f"  Azure deployment name: {os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME', 'NOT SET')}")
print(f"  API endpoint set: {'YES' if os.environ.get('AZURE_OPENAI_ENDPOINT') else 'NO'}")
print(f"  API key set: {'YES' if os.environ.get('AZURE_OPENAI_API_KEY') else 'NO'}")
print()

print("=== Testing LLM Connection ===")
# Test if LLM is working
try:
    test_response = llm.invoke("Say hello")
    print(f"LLM connection test: ✓ Success")
    print(f"  Response: {test_response.content[:50]}...")
except Exception as e:
    print(f"LLM connection test: ✗ Failed")
    print(f"  Error: {e}")
print()

print("Tutorial complete! You've learned debugging strategies for LangChain.")

