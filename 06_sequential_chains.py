"""
06_sequential_chains.py - Sequential Chains

This tutorial demonstrates:
- Sequential chain composition
- Passing data between chains
- Multi-step workflows
- Chain dependencies

Prerequisites:
- Understanding of basic chains
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import LLMChain, SimpleSequentialChain, SequentialChain

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
)

print("=== Simple Sequential Chain ===")
# SimpleSequentialChain: Output of one chain is input to the next
# Useful when each step uses the output of the previous step directly

# Step 1: Generate a story concept
story_prompt = ChatPromptTemplate.from_messages([
    ("human", "Come up with a creative story concept about {topic}."),
])
story_chain = LLMChain(llm=llm, prompt=story_prompt, output_key="story_concept")

# Step 2: Expand the concept into a full story
expand_prompt = ChatPromptTemplate.from_messages([
    ("human", "Expand this story concept into a short story:\n\n{story_concept}"),
])
expand_chain = LLMChain(llm=llm, prompt=expand_prompt, output_key="story")

# Combine into a sequential chain
sequential_chain = SimpleSequentialChain(
    chains=[story_chain, expand_chain],
    verbose=True  # Set to True to see intermediate outputs
)

result = sequential_chain.run("a robot learning to paint")
print(f"\nFinal Story:\n{result}\n")

print("=== Sequential Chain with Named Outputs ===")
# SequentialChain allows you to name outputs and use them selectively
# This is more flexible than SimpleSequentialChain

# Chain 1: Generate product name
name_prompt = ChatPromptTemplate.from_messages([
    ("human", "Come up with a creative name for a {product_type} product."),
])
name_chain = LLMChain(
    llm=llm, 
    prompt=name_prompt, 
    output_key="product_name"
)

# Chain 2: Generate product description
desc_prompt = ChatPromptTemplate.from_messages([
    ("human", "Write a compelling description for a product named '{product_name}' of type {product_type}."),
])
desc_chain = LLMChain(
    llm=llm, 
    prompt=desc_prompt, 
    output_key="product_description"
)

# Chain 3: Generate marketing tagline
tagline_prompt = ChatPromptTemplate.from_messages([
    ("human", "Create a catchy marketing tagline for:\nProduct: {product_name}\nDescription: {product_description}"),
])
tagline_chain = LLMChain(
    llm=llm, 
    prompt=tagline_prompt, 
    output_key="tagline"
)

# Combine with SequentialChain (allows selective input/output usage)
marketing_chain = SequentialChain(
    chains=[name_chain, desc_chain, tagline_chain],
    input_variables=["product_type"],  # Initial input
    output_variables=["product_name", "product_description", "tagline"],  # Final outputs
    verbose=True
)

result = marketing_chain({"product_type": "smartwatch"})
print(f"\nProduct Name: {result['product_name']}")
print(f"Description: {result['product_description']}")
print(f"Tagline: {result['tagline']}\n")

print("=== Multi-Step Analysis Chain ===")
# Create a chain that analyzes code in multiple steps

# Step 1: Analyze code structure
structure_prompt = ChatPromptTemplate.from_messages([
    ("human", "Analyze the structure of this code:\n\n{code}\n\nFocus on architecture and organization."),
])
structure_chain = LLMChain(llm=llm, prompt=structure_prompt, output_key="structure_analysis")

# Step 2: Analyze code quality
quality_prompt = ChatPromptTemplate.from_messages([
    ("human", "Based on this code:\n\n{code}\n\nAnd this structure analysis:\n\n{structure_analysis}\n\nEvaluate code quality, best practices, and potential issues."),
])
quality_chain = LLMChain(llm=llm, prompt=quality_prompt, output_key="quality_analysis")

# Step 3: Generate recommendations
recommendations_prompt = ChatPromptTemplate.from_messages([
    ("human", "Based on the code analysis:\n{quality_analysis}\n\nProvide specific improvement recommendations."),
])
recommendations_chain = LLMChain(llm=llm, prompt=recommendations_prompt, output_key="recommendations")

code_analysis_chain = SequentialChain(
    chains=[structure_chain, quality_chain, recommendations_chain],
    input_variables=["code"],
    output_variables=["structure_analysis", "quality_analysis", "recommendations"],
    verbose=True
)

sample_code = """
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
"""

result = code_analysis_chain({"code": sample_code})
print(f"\nStructure Analysis:\n{result['structure_analysis']}\n")
print(f"Quality Analysis:\n{result['quality_analysis']}\n")
print(f"Recommendations:\n{result['recommendations']}\n")

print("=== Chain Pipeline Example ===")
# Create a pipeline for content generation and refinement

# Step 1: Generate initial content
generate_prompt = ChatPromptTemplate.from_messages([
    ("human", "Write a brief article about {topic}."),
])
generate_chain = LLMChain(llm=llm, prompt=generate_prompt, output_key="initial_content")

# Step 2: Improve writing style
improve_prompt = ChatPromptTemplate.from_messages([
    ("human", "Improve the writing style of this article while keeping the content:\n\n{initial_content}"),
])
improve_chain = LLMChain(llm=llm, prompt=improve_prompt, output_key="improved_content")

# Step 3: Add conclusion
conclusion_prompt = ChatPromptTemplate.from_messages([
    ("human", "Add a strong conclusion to this article:\n\n{improved_content}"),
])
conclusion_chain = LLMChain(llm=llm, prompt=conclusion_prompt, output_key="final_content")

content_pipeline = SequentialChain(
    chains=[generate_chain, improve_chain, conclusion_chain],
    input_variables=["topic"],
    output_variables=["final_content"],
    verbose=True
)

result = content_pipeline({"topic": "renewable energy benefits"})
print(f"\nFinal Article:\n{result['final_content']}\n")

print("Tutorial complete! You've learned how to create sequential chains for complex workflows.")

