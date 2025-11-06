"""
21_callbacks.py - Callbacks and Monitoring

This tutorial demonstrates:
- Custom callbacks
- Logging and monitoring
- Performance tracking
- Debugging with callbacks

Prerequisites:
- Basic LangChain understanding
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks import StdOutCallbackHandler
from typing import Any, Dict, List
from langchain_classic.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
)

print("=== What are Callbacks? ===")
print("""
Callbacks allow you to:
- Monitor chain execution
- Log events and actions
- Track performance metrics
- Debug issues
- Integrate with monitoring systems
""")

print("=== Standard Output Callback ===")
# StdOutCallbackHandler prints execution details
stdout_handler = StdOutCallbackHandler()

llm_with_callback = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
    callbacks=[stdout_handler],
)

print("Using StdOutCallbackHandler:")
response = llm_with_callback.invoke("What is Python?")
print(f"Response: {response.content}\n")

print("=== Custom Callback for Logging ===")
# Create a custom callback that logs events
class LoggingCallbackHandler(BaseCallbackHandler):
    """Logs chain execution events"""
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Called when LLM starts"""
        print(f"[LOG] LLM started with {len(prompts)} prompt(s)")
        print(f"[LOG] First prompt preview: {prompts[0][:50]}...")
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Called when LLM finishes"""
        print(f"[LOG] LLM completed")
        if hasattr(response, 'llm_output'):
            print(f"[LOG] Token usage info available: {bool(response.llm_output)}")
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when LLM errors"""
        print(f"[LOG] LLM error: {error}")
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> None:
        """Called when chain starts"""
        print(f"[LOG] Chain started: {serialized.get('name', 'Unknown')}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Called when chain ends"""
        print(f"[LOG] Chain completed")
    
    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when chain errors"""
        print(f"[LOG] Chain error: {error}")

logging_handler = LoggingCallbackHandler()

print("Custom logging callback example:")
logging_llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
    callbacks=[logging_handler],
)

response = logging_llm.invoke("Explain machine learning briefly.")
print(f"\nResponse: {response.content}\n")

print("=== Performance Tracking Callback ===")
# Track performance metrics
import time

class PerformanceCallbackHandler(BaseCallbackHandler):
    """Tracks performance metrics"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.token_count = 0
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Record start time"""
        self.start_time = time.time()
        # Estimate token count (rough approximation: 1 token ≈ 4 chars)
        total_chars = sum(len(p) for p in prompts)
        self.token_count = total_chars // 4
    
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Record end time and calculate metrics"""
        self.end_time = time.time()
        if self.start_time:
            duration = self.end_time - self.start_time
            print(f"\n[PERFORMANCE] Duration: {duration:.2f} seconds")
            print(f"[PERFORMANCE] Estimated tokens: {self.token_count}")
            if duration > 0:
                print(f"[PERFORMANCE] Tokens/second: {self.token_count / duration:.2f}")

performance_handler = PerformanceCallbackHandler()

print("Performance tracking example:")
perf_llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
    callbacks=[performance_handler],
)

response = perf_llm.invoke("Write a detailed explanation of neural networks.")
print(f"\nResponse: {response.content[:100]}...\n")

print("=== Multiple Callbacks ===")
# Use multiple callbacks together
print("Using multiple callbacks:")

multiple_callbacks = [
    logging_handler,
    performance_handler,
]

multi_llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
    callbacks=multiple_callbacks,
)

response = multi_llm.invoke("What is LangChain?")
print(f"\nResponse: {response.content}\n")

print("=== Callbacks with Chains ===")
# Callbacks work with chains
prompt = ChatPromptTemplate.from_messages([
    ("human", "Explain {topic} in {style} style."),
])

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    callbacks=[logging_handler],  # Add callback to chain
)

print("Chain with callback:")
result = chain.run(topic="Python programming", style="simple")
print(f"\nResult: {result}\n")

print("=== Error Handling Callback ===")
# Callback for error tracking
class ErrorTrackingCallback(BaseCallbackHandler):
    """Tracks errors for monitoring"""
    
    def __init__(self):
        self.errors = []
    
    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Track LLM errors"""
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": time.time(),
        }
        self.errors.append(error_info)
        print(f"[ERROR] LLM error: {error_info}")
    
    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """Track chain errors"""
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": time.time(),
        }
        self.errors.append(error_info)
        print(f"[ERROR] Chain error: {error_info}")
    
    def get_error_count(self) -> int:
        """Get total error count"""
        return len(self.errors)

error_handler = ErrorTrackingCallback()

error_tracking_llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
    callbacks=[error_handler],
)

# Normal operation
response = error_tracking_llm.invoke("Hello!")
print(f"Errors tracked: {error_handler.get_error_count()}\n")

print("=== Callback Best Practices ===")
print("""
1. Use for debugging:
   - Set verbose=False and use callbacks for logging
   - Track execution flow
   - Monitor performance

2. Production monitoring:
   - Log to external systems
   - Track metrics and errors
   - Monitor token usage

3. Multiple callbacks:
   - Combine different callback types
   - Separate concerns (logging, metrics, errors)

4. Performance:
   - Keep callbacks lightweight
   - Use async for external logging if needed

5. Error handling:
   - Handle callback errors gracefully
   - Don't let callbacks break your chain
""")

print("Tutorial complete! You've learned how to use callbacks for monitoring and debugging.")

