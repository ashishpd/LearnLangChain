"""
23_custom_chains.py - Building Custom Chains

This tutorial demonstrates:
- Creating custom chain classes
- Chain composition
- Reusable chain components
- Custom chain logic

Prerequisites:
- Understanding of chains and LangChain architecture
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.chains.base import Chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from typing import Dict, List, Optional

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
)

print("=== What are Custom Chains? ===")
print("""
Custom chains allow you to:
- Create reusable chain logic
- Encapsulate complex workflows
- Add custom processing steps
- Build domain-specific chains
""")

print("=== Simple Custom Chain ===")
# Create a basic custom chain by extending Chain class

class TextAnalysisChain(Chain):
    """Custom chain for text analysis"""
    
    # Define input/output variables
    input_key: str = "text"
    output_key: str = "analysis"
    
    # Sub-chains
    summary_chain: LLMChain
    sentiment_chain: LLMChain
    
    @property
    def input_keys(self) -> List[str]:
        """Define input keys"""
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        """Define output keys"""
        return [self.output_key]
    
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Execute the chain"""
        text = inputs[self.input_key]
        
        # Run sub-chains
        summary = self.summary_chain.run(text=text)
        sentiment = self.sentiment_chain.run(text=text)
        
        # Combine results
        analysis = f"Summary: {summary}\nSentiment: {sentiment}"
        
        return {self.output_key: analysis}

# Create sub-chains
summary_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_messages([
        ("human", "Summarize in one sentence: {text}"),
    ]),
)

sentiment_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_messages([
        ("human", "Analyze sentiment: {text}"),
    ]),
)

# Create custom chain
analysis_chain = TextAnalysisChain(
    summary_chain=summary_chain,
    sentiment_chain=sentiment_chain,
)

print("Testing custom TextAnalysisChain:")
result = analysis_chain.run("LangChain is an excellent framework for building AI applications.")
print(f"Result: {result}\n")

print("=== Advanced Custom Chain with Processing ===")
# Chain with custom processing logic

class CodeReviewChain(Chain):
    """Custom chain for code review"""
    
    input_key: str = "code"
    language_key: str = "language"
    output_key: str = "review"
    
    # Sub-chains
    structure_chain: LLMChain
    quality_chain: LLMChain
    suggestions_chain: LLMChain
    
    def _preprocess_code(self, code: str) -> str:
        """Preprocess code before analysis"""
        # Remove extra whitespace, normalize
        return code.strip()
    
    def _format_review(self, structure: str, quality: str, suggestions: str) -> str:
        """Format the final review"""
        return f"""Code Review Report:
==================
Structure Analysis:
{structure}

Quality Assessment:
{quality}

Suggestions:
{suggestions}
"""
    
    @property
    def input_keys(self) -> List[str]:
        return [self.input_key, self.language_key]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Execute code review chain"""
        code = self._preprocess_code(inputs[self.input_key])
        language = inputs[self.language_key]
        
        # Run analysis chains
        structure = self.structure_chain.run(code=code, language=language)
        quality = self.quality_chain.run(code=code)
        suggestions = self.suggestions_chain.run(code=code)
        
        # Format final output
        review = self._format_review(structure, quality, suggestions)
        
        return {self.output_key: review}

# Create sub-chains for code review
structure_prompt = ChatPromptTemplate.from_messages([
    ("human", "Analyze the structure of this {language} code:\n\n{code}"),
])
structure_chain = LLMChain(llm=llm, prompt=structure_prompt)

quality_prompt = ChatPromptTemplate.from_messages([
    ("human", "Assess code quality for:\n\n{code}"),
])
quality_chain = LLMChain(llm=llm, prompt=quality_prompt)

suggestions_prompt = ChatPromptTemplate.from_messages([
    ("human", "Provide improvement suggestions for:\n\n{code}"),
])
suggestions_chain = LLMChain(llm=llm, prompt=suggestions_prompt)

# Create code review chain
code_review_chain = CodeReviewChain(
    structure_chain=structure_chain,
    quality_chain=quality_chain,
    suggestions_chain=suggestions_chain,
)

sample_code = """
def add_numbers(a, b):
    return a + b
"""

print("Testing CodeReviewChain:")
result = code_review_chain.run(
    code=sample_code,
    language="Python"
)
print(result)
print()

print("=== Chain with Validation ===")
# Custom chain with input validation

class ValidatedChain(Chain):
    """Chain with input validation"""
    
    input_key: str = "input"
    output_key: str = "output"
    
    llm_chain: LLMChain
    
    def _validate_input(self, inputs: Dict[str, str]) -> bool:
        """Validate input"""
        if self.input_key not in inputs:
            return False
        if len(inputs[self.input_key]) < 10:
            print("Warning: Input too short")
            return False
        return True
    
    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Execute with validation"""
        if not self._validate_input(inputs):
            return {self.output_key: "Error: Invalid input"}
        
        result = self.llm_chain.run(**inputs)
        return {self.output_key: result}

# Create validated chain
prompt = ChatPromptTemplate.from_messages([
    ("human", "Process: {input}"),
])
llm_chain = LLMChain(llm=llm, prompt=prompt)

validated = ValidatedChain(llm_chain=llm_chain)

print("Testing ValidatedChain:")
result = validated.run(input="This is a valid input with enough characters.")
print(f"Result: {result}\n")

result = validated.run(input="Short")
print(f"Result with short input: {result}\n")

print("=== Composable Chain Pattern ===")
# Chain that can be composed with others

class PreprocessChain(Chain):
    """Preprocessing chain that can be composed"""
    
    input_key: str = "raw_text"
    output_key: str = "processed_text"
    
    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]
    
    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Preprocess text"""
        text = inputs[self.input_key]
        # Simple preprocessing
        processed = text.strip().title()
        return {self.output_key: processed}

preprocess = PreprocessChain()

# Create analysis chain that uses preprocessed text
analysis_prompt = ChatPromptTemplate.from_messages([
    ("human", "Analyze: {processed_text}"),
])
analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt)

# Compose them
from langchain.chains import SequentialChain

composed = SequentialChain(
    chains=[preprocess, analysis_chain],
    input_variables=["raw_text"],
    output_variables=["text"],  # Output from analysis_chain
)

print("Composed chain example:")
result = composed.run(raw_text="  langchain is great  ")
print(f"Result: {result}\n")

print("=== Best Practices for Custom Chains ===")
print("""
1. Clear Interface:
   - Define input_keys and output_keys clearly
   - Document expected inputs/outputs
   - Use type hints

2. Modularity:
   - Build from smaller chains
   - Reuse existing chains
   - Single responsibility

3. Error Handling:
   - Validate inputs
   - Handle errors gracefully
   - Provide useful error messages

4. Testing:
   - Test individual components
   - Test chain as a whole
   - Test edge cases

5. Documentation:
   - Document chain purpose
   - Explain inputs/outputs
   - Provide usage examples
""")

print("Tutorial complete! You've learned how to build custom chains.")

