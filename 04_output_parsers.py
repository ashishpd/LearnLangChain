"""
04_output_parsers.py - Output Parsers

This tutorial demonstrates:
- Parsing structured outputs from LLMs
- Using Pydantic models for type-safe outputs
- JSON parsing
- Error handling for parsing failures
- Custom output formats

Prerequisites:
- Azure OpenAI setup
- Understanding of prompt templates
"""

import os
import json
from typing import List
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers.json import SimpleJsonOutputParser
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
)

print("=== Simple JSON Parser ===")
# Parse JSON output from LLM
json_parser = SimpleJsonOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that returns JSON responses."),
    ("human", "List 3 programming languages with their primary use case. Return as JSON array."),
    ("system", "Format: {format_instructions}"),
]).partial(format_instructions=json_parser.get_format_instructions())

response = llm.invoke(prompt.format_messages())
parsed = json_parser.parse(response.content)
print(f"Parsed JSON: {parsed}\n")

print("=== Pydantic Model Parser ===")
# Define a structured output using Pydantic
class ProgrammingLanguage(BaseModel):
    """Information about a programming language"""
    name: str = Field(description="Name of the programming language")
    year_created: int = Field(description="Year the language was created")
    primary_use: str = Field(description="Primary use case of the language")
    popularity_rank: int = Field(description="Current popularity ranking (1-10)")

class LanguageList(BaseModel):
    """List of programming languages"""
    languages: List[ProgrammingLanguage] = Field(description="List of programming languages")
    total_count: int = Field(description="Total number of languages listed")

# Create parser with Pydantic model
pydantic_parser = PydanticOutputParser(pydantic_object=LanguageList)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a programming language expert. {format_instructions}"),
    ("human", "Provide information about 3 popular programming languages."),
]).partial(format_instructions=pydantic_parser.get_format_instructions())

response = llm.invoke(prompt.format_messages())
parsed_data = pydantic_parser.parse(response.content)
print(f"Parsed Pydantic object:")
print(f"Total count: {parsed_data.total_count}")
for lang in parsed_data.languages:
    print(f"  - {lang.name} ({lang.year_created}): {lang.primary_use} - Rank: {lang.popularity_rank}")
print()

print("=== Error Handling in Parsing ===")
# When parsing fails, you can catch exceptions and handle them
try:
    # Attempt to parse a response
    prompt_with_error = ChatPromptTemplate.from_messages([
        ("system", "Return data about a book. {format_instructions}"),
        ("human", "Tell me about the book '1984' by George Orwell."),
    ]).partial(format_instructions=pydantic_parser.get_format_instructions())
    
    response = llm.invoke(prompt_with_error.format_messages())
    # Try to parse - if it fails, we catch the exception
    try:
        parsed = pydantic_parser.parse(response.content)
        print(f"Successfully parsed: {parsed}\n")
    except Exception as e:
        print(f"Parsing error occurred: {type(e).__name__}")
        print(f"Response content: {response.content[:200]}...")
        print("Note: In production, you could use error recovery strategies here.\n")
except Exception as e:
    print(f"Error during parsing demonstration: {e}\n")

print("=== Custom Parser for Specific Format ===")
# Create a custom parser for a specific format
class CustomListParser:
    """Parser for numbered list format"""
    
    def parse(self, text: str) -> List[dict]:
        """Parse numbered list into list of dictionaries"""
        items = []
        lines = text.strip().split('\n')
        for line in lines:
            if line.strip() and (line[0].isdigit() or line.startswith('-')):
                # Extract text after number/dash
                text_part = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                items.append({"item": text_part})
        return items

custom_parser = CustomListParser()
prompt = ChatPromptTemplate.from_messages([
    ("system", "Return a numbered list."),
    ("human", "List 5 benefits of cloud computing. Use numbered format."),
])

response = llm.invoke(prompt.format_messages())
parsed_list = custom_parser.parse(response.content)
print(f"Parsed list: {parsed_list}\n")

print("=== Complex Nested Parser ===")
# Parse complex nested structures
class Address(BaseModel):
    street: str
    city: str
    country: str

class Person(BaseModel):
    name: str
    age: int
    email: str
    address: Address
    hobbies: List[str]

person_parser = PydanticOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Create a person profile. {format_instructions}"),
    ("human", "Generate a profile for a fictional character named Alex who loves technology."),
]).partial(format_instructions=person_parser.get_format_instructions())

response = llm.invoke(prompt.format_messages())
person_data = person_parser.parse(response.content)
print(f"Parsed Person:")
print(f"  Name: {person_data.name}")
print(f"  Age: {person_data.age}")
print(f"  Email: {person_data.email}")
print(f"  Address: {person_data.address.street}, {person_data.address.city}, {person_data.address.country}")
print(f"  Hobbies: {', '.join(person_data.hobbies)}\n")

print("Tutorial complete! You've learned how to parse and structure LLM outputs.")

