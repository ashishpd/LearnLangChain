"""
07_router_chains.py - Router Chains

This tutorial demonstrates:
- Router chains for conditional logic
- Multi-route processing
- Routing based on input content
- Branching workflows

Prerequisites:
- Understanding of sequential chains
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_classic.chains import LLMChain, SequentialChain
from langchain_classic.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain_classic.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
)

print("=== Manual Routing with Conditional Logic ===")
# Simple routing based on conditions

# Define different chains for different purposes
technical_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical expert. Provide detailed technical explanations."),
    ("human", "{query}"),
])
technical_chain = LLMChain(llm=llm, prompt=technical_prompt)

simple_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a teacher. Explain concepts simply for beginners."),
    ("human", "{query}"),
])
simple_chain = LLMChain(llm=llm, prompt=simple_prompt)

creative_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative writer. Provide engaging, creative responses."),
    ("human", "{query}"),
])
creative_chain = LLMChain(llm=llm, prompt=creative_prompt)

def route_query(query: str, style: str = "auto") -> str:
    """Route query to appropriate chain based on style"""
    if style == "technical":
        return technical_chain.run(query)
    elif style == "simple":
        return simple_chain.run(query)
    elif style == "creative":
        return creative_chain.run(query)
    else:
        # Auto-detect based on keywords
        if any(word in query.lower() for word in ["explain", "how does", "what is", "define"]):
            return simple_chain.run(query)
        elif any(word in query.lower() for word in ["architecture", "implementation", "algorithm"]):
            return technical_chain.run(query)
        else:
            return creative_chain.run(query)

print("Routing 'Explain quantum computing':")
result = route_query("Explain quantum computing", style="simple")
print(f"Result: {result}\n")

print("Routing 'Design a distributed system architecture':")
result = route_query("Design a distributed system architecture", style="technical")
print(f"Result: {result}\n")

print("=== Multi-Prompt Router Chain ===")
# LangChain provides LLMRouterChain for intelligent routing
# Define destination chains
destinations = [
    "Technical Support: Best for technical questions, debugging, and implementation details",
    "General Knowledge: Best for general questions, explanations, and educational content",
    "Creative Writing: Best for creative tasks, stories, and imaginative content",
]

destination_chains = {
    "Technical Support": technical_chain,
    "General Knowledge": simple_chain,
    "Creative Writing": creative_chain,
}

# Create router prompt
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

router_chain = LLMRouterChain.from_llm(llm, router_prompt)

def route_with_llm(query: str):
    """Route query using LLM router"""
    # Get routing decision
    routing_decision = router_chain.route(query)
    destination = routing_decision["next_inputs"]["destination"]
    
    print(f"Query: {query}")
    print(f"Routed to: {destination}")
    
    # Execute the appropriate chain
    if destination in destination_chains:
        result = destination_chains[destination].run(query)
        print(f"Response: {result}\n")
        return result
    else:
        return "Destination not found"

print("Testing LLM-based routing:")
route_with_llm("How do I fix a Python import error?")
route_with_llm("Tell me about the history of computers")
route_with_llm("Write a short poem about AI")

print("=== Advanced Routing with Multiple Criteria ===")
# Route based on multiple factors

code_review_prompt = ChatPromptTemplate.from_messages([
    ("human", "Review this {language} code:\n\n{code}"),
])
code_review_chain = LLMChain(llm=llm, prompt=code_review_prompt)

explanation_prompt = ChatPromptTemplate.from_messages([
    ("human", "Explain {concept} in {style} style."),
])
explanation_chain = LLMChain(llm=llm, prompt=explanation_prompt)

def smart_router(query_type: str, **kwargs):
    """Route based on query type and parameters"""
    if query_type == "code_review":
        return code_review_chain.run(
            language=kwargs.get("language", "Python"),
            code=kwargs.get("code", "")
        )
    elif query_type == "explanation":
        return explanation_chain.run(
            concept=kwargs.get("concept", ""),
            style=kwargs.get("style", "simple")
        )
    elif query_type == "technical":
        return technical_chain.run(kwargs.get("query", ""))
    else:
        return simple_chain.run(kwargs.get("query", ""))

print("Advanced routing examples:")
print("\n1. Code Review:")
result = smart_router(
    "code_review",
    language="Python",
    code="def add(a, b): return a + b"
)
print(f"{result}\n")

print("2. Explanation:")
result = smart_router(
    "explanation",
    concept="machine learning",
    style="technical"
)
print(f"{result}\n")

print("=== Sequential Chain with Routing ===")
# Combine routing with sequential processing

# Step 1: Determine query type
classify_prompt = ChatPromptTemplate.from_messages([
    ("human", "Classify this query as one of: code, explanation, creative.\n\nQuery: {query}"),
])
classify_chain = LLMChain(llm=llm, prompt=classify_prompt, output_key="query_type")

# Step 2: Route based on type and process
# This is a simplified example - in practice you'd use the classification result
routing_chain = SequentialChain(
    chains=[classify_chain],
    input_variables=["query"],
    output_variables=["query_type"],
    verbose=True
)

result = routing_chain({"query": "How does recursion work in programming?"})
print(f"Classified as: {result['query_type']}\n")

print("Tutorial complete! You've learned about routing and conditional chain execution.")

