"""
24_async_operations.py - Async Operations

This tutorial demonstrates:
- Async/await patterns with LangChain
- Concurrent processing
- Performance optimization
- Async LLM calls

Prerequisites:
- Understanding of Python async/await
- Basic LangChain knowledge
"""

import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import time

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
)

print("=== Why Async? ===")
print("""
Async operations allow:
- Concurrent execution
- Better resource utilization
- Improved performance for multiple requests
- Non-blocking I/O operations
""")

print("=== Synchronous vs Async ===")
# Compare synchronous vs async execution

prompt = ChatPromptTemplate.from_messages([
    ("human", "Say hello and mention: {topic}"),
])

print("=== Synchronous Execution ===")
start_time = time.time()

results_sync = []
topics = ["Python", "AI", "LangChain"]

for topic in topics:
    result = llm.invoke(prompt.format_messages(topic=topic))
    results_sync.append(result.content)
    print(f"  Completed: {topic}")

sync_duration = time.time() - start_time
print(f"Synchronous duration: {sync_duration:.2f} seconds\n")

print("=== Async Execution ===")
# Async version
async def async_llm_call(topic: str):
    """Async LLM call"""
    return await llm.ainvoke(prompt.format_messages(topic=topic))

async def run_async_example():
    """Run async example"""
    start_time = time.time()
    
    # Create tasks for concurrent execution
    tasks = [async_llm_call(topic) for topic in topics]
    
    # Execute concurrently
    results_async = await asyncio.gather(*tasks)
    
    async_duration = time.time() - start_time
    
    print("Async results:")
    for topic, result in zip(topics, results_async):
        print(f"  Completed: {topic}")
    
    print(f"Async duration: {async_duration:.2f} seconds")
    print(f"Speedup: {sync_duration / async_duration:.2f}x\n")
    
    return results_async

# Run async example
results_async = asyncio.run(run_async_example())

print("=== Batch Async Processing ===")
# Process multiple items concurrently

async def process_multiple_queries():
    """Process multiple queries concurrently"""
    queries = [
        "What is Python?",
        "What is machine learning?",
        "What is LangChain?",
        "What are embeddings?",
    ]
    
    start_time = time.time()
    
    # Create async tasks
    tasks = [llm.ainvoke(query) for query in queries]
    
    # Execute all concurrently
    responses = await asyncio.gather(*tasks)
    
    duration = time.time() - start_time
    
    print(f"Processed {len(queries)} queries in {duration:.2f} seconds")
    print("Responses:")
    for query, response in zip(queries, responses):
        print(f"  Q: {query}")
        print(f"  A: {response.content[:60]}...")
    print()
    
    return responses

asyncio.run(process_multiple_queries())

print("=== Async with Chains ===")
# Async chains

from langchain.chains import LLMChain

chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_messages([
        ("human", "Explain {concept} in {style} style."),
    ]),
)

async def run_chain_async():
    """Run chain async"""
    concepts = ["Python", "AI", "Cloud"]
    styles = ["simple", "technical", "creative"]
    
    start_time = time.time()
    
    # Create async tasks for chains
    tasks = [
        chain.ainvoke({"concept": c, "style": s})
        for c, s in zip(concepts, styles)
    ]
    
    results = await asyncio.gather(*tasks)
    
    duration = time.time() - start_time
    
    print(f"Processed {len(concepts)} chains in {duration:.2f} seconds")
    for concept, result in zip(concepts, results):
        print(f"  {concept}: {result['text'][:50]}...")
    print()

asyncio.run(run_chain_async())

print("=== Async with Error Handling ===")
# Handle errors in async operations

async def safe_async_call(query: str):
    """Async call with error handling"""
    try:
        response = await llm.ainvoke(query)
        return {"success": True, "response": response.content}
    except Exception as e:
        return {"success": False, "error": str(e)}

async def process_with_errors():
    """Process multiple queries with error handling"""
    queries = [
        "What is Python?",
        "Invalid query that might fail",
        "What is AI?",
    ]
    
    tasks = [safe_async_call(q) for q in queries]
    results = await asyncio.gather(*tasks)
    
    print("Results with error handling:")
    for query, result in zip(queries, results):
        if result["success"]:
            print(f"  ✓ {query}: {result['response'][:50]}...")
        else:
            print(f"  ✗ {query}: Error - {result['error']}")
    print()

asyncio.run(process_with_errors())

print("=== Async Best Practices ===")
print("""
1. Use async for I/O-bound operations
   - API calls
   - Database queries
   - File operations

2. Batch processing:
   - Process multiple items concurrently
   - Use asyncio.gather() for parallel execution
   - Set reasonable limits with Semaphore

3. Error handling:
   - Handle errors in async functions
   - Use try-except in async contexts
   - Consider retry logic

4. Resource management:
   - Limit concurrency with Semaphore
   - Close resources properly
   - Monitor resource usage

5. Testing:
   - Test async code with pytest-asyncio
   - Mock async functions appropriately
   - Test error scenarios
""")

print("=== Performance Considerations ===")
print("""
Async benefits:
- Significant speedup for multiple operations
- Better resource utilization
- Scalable for high concurrency

When to use:
- Processing multiple items
- I/O-bound operations
- High-throughput scenarios

When not needed:
- Single operations
- CPU-bound tasks
- Simple scripts
""")

print("Tutorial complete! You've learned about async operations with LangChain.")

