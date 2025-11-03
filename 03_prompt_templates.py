"""
03_prompt_templates.py - Prompt Templates

This tutorial demonstrates:
- Creating reusable prompt templates
- Variable substitution in prompts
- Different template formatting methods
- Building dynamic prompts

Prerequisites:
- Azure OpenAI setup from previous tutorials
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
)

print("=== Simple Prompt Template ===")
# Create a simple prompt template with variables
template = "Tell me about {topic} in {language} language."
prompt_template = PromptTemplate.from_template(template)

# Format the template with variables
prompt = prompt_template.format(topic="artificial intelligence", language="simple")
print(f"Formatted prompt: {prompt}\n")
response = llm.invoke(prompt)
print(f"Response: {response.content}\n")

print("=== Chat Prompt Template (Structured) ===")
# Chat prompt templates are better for conversational AI
# They separate system and human messages
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains concepts clearly."),
    ("human", "Explain {concept} to someone who is new to {field}."),
])

prompt = chat_template.format_messages(
    concept="neural networks",
    field="machine learning"
)
print(f"System message: {prompt[0].content}")
print(f"Human message: {prompt[1].content}\n")
response = llm.invoke(prompt)
print(f"Response: {response.content}\n")

print("=== Multi-Variable Template ===")
# Templates can have multiple variables
multi_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} who provides {style} explanations."),
    ("human", "Explain {topic} using {examples} examples."),
])

prompt = multi_template.format_messages(
    role="science teacher",
    style="clear and engaging",
    topic="photosynthesis",
    examples="3"
)
response = llm.invoke(prompt)
print(f"Response: {response.content}\n")

print("=== Template with Examples ===")
# Templates can include structured examples
example_template = ChatPromptTemplate.from_messages([
    ("system", """You are a code reviewer. Review the following {language} code and provide feedback.
    Format your response with:
    1. Code quality score (1-10)
    2. Issues found
    3. Suggestions for improvement"""),
    ("human", "Code:\n{code}"),
])

code_sample = """
def add_numbers(a, b):
    return a + b

result = add_numbers(5, 10)
print(result)
"""

prompt = example_template.format_messages(
    language="Python",
    code=code_sample
)
response = llm.invoke(prompt)
print(f"Response: {response.content}\n")

print("=== Dynamic Template Building ===")
# You can build templates programmatically
def create_review_template(product_type: str) -> ChatPromptTemplate:
    """Create a product review template dynamically"""
    return ChatPromptTemplate.from_messages([
        ("system", f"You are an expert reviewer of {product_type}."),
        ("human", "Review this {product_type}:\n{description}\n\nProvide pros and cons."),
    ])

# Use the dynamic template
review_template = create_review_template("smartphone")
prompt = review_template.format_messages(
    description="A phone with excellent camera, long battery life, but high price"
)
response = llm.invoke(prompt)
print(f"Response: {response.content}\n")

print("Tutorial complete! You've learned how to create and use prompt templates.")

