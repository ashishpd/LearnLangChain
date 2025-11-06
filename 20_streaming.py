"""
20_streaming.py - Streaming Responses

This tutorial demonstrates:
- Streaming LLM responses
- Real-time output generation
- Callback handlers for streaming
- User experience improvements

Prerequisites:
- Azure OpenAI setup
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage

# Load environment variables
load_dotenv()

# Initialize LLM with streaming
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
    streaming=True,  # Enable streaming
)

print("=== What is Streaming? ===")
print("""
Streaming allows:
- Real-time output as LLM generates tokens
- Better user experience (no waiting for complete response)
- Perceived faster responses
- Progressive content display
""")

print("=== Basic Streaming ===")
# Streaming with StreamingStdOutCallbackHandler
# This prints tokens as they're generated

print("\nStreaming response:")
streaming_llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

# Note: StreamingStdOutCallbackHandler prints directly to stdout
# So the response appears as it's generated
response = streaming_llm.invoke(
    "Write a short story about a robot learning to code (2-3 sentences)."
)
print("\n\nStreaming complete!\n")

print("=== Custom Streaming Handler ===")
# Create a custom callback for more control
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any

class CustomStreamHandler(BaseCallbackHandler):
    """Custom handler that collects tokens"""
    
    def __init__(self):
        self.tokens = []
        self.complete_text = ""
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called when a new token is generated"""
        self.tokens.append(token)
        self.complete_text += token
        # Print token as it arrives (you could do other things here)
        print(token, end="", flush=True)
    
    def get_complete_text(self) -> str:
        """Get the complete generated text"""
        return self.complete_text

custom_handler = CustomStreamHandler()
streaming_llm_custom = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
    streaming=True,
    callbacks=[custom_handler],
)

print("\nCustom streaming handler example:")
response = streaming_llm_custom.invoke(
    "Explain what streaming is in one sentence."
)
print("\n")
print(f"Complete text collected: {custom_handler.get_complete_text()}\n")

print("=== Streaming with Chains ===")
# Streaming works with chains too
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("human", "Write a {length} explanation about {topic}."),
])

chain = prompt | streaming_llm_custom | StrOutputParser()

print("Streaming chain example:")
result = chain.invoke({"length": "brief", "topic": "artificial intelligence"})
print("\n")

print("=== Streaming Best Practices ===")
print("""
1. Enable streaming for better UX
   - Users see progress immediately
   - Especially important for long responses

2. Handle partial responses
   - Display tokens as they arrive
   - Update UI progressively

3. Error handling
   - Handle streaming errors gracefully
   - Provide fallback to non-streaming

4. Buffer management
   - Consider buffering for better display
   - Handle token boundaries

5. Performance
   - Streaming can improve perceived performance
   - Use for interactive applications
""")

print("=== Non-Streaming Comparison ===")
# Compare with non-streaming
non_streaming_llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
    streaming=False,  # Explicitly disable streaming
)

print("Non-streaming (waits for complete response):")
response = non_streaming_llm.invoke("Say hello in a creative way.")
print(f"Response: {response.content}\n")

print("=== Streaming Use Cases ===")
print("""
Good for:
- Interactive chatbots
- Long-form content generation
- Real-time applications
- Better user experience

Consider non-streaming for:
- Batch processing
- When you need complete response first
- Simple one-off queries
""")

print("Tutorial complete! You've learned about streaming responses in LangChain.")

