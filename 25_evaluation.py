"""
25_evaluation.py - Evaluating LLM Outputs

This tutorial demonstrates:
- Evaluating LLM outputs
- Metrics and benchmarks
- Quality assessment
- Testing strategies

Prerequisites:
- Basic LangChain understanding
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    temperature=0.7,
)

print("=== Why Evaluate LLM Outputs? ===")
print("""
Evaluation helps:
- Ensure quality and accuracy
- Compare different models/configurations
- Monitor performance over time
- Identify areas for improvement
""")

print("=== Simple Evaluation Functions ===")
# Create evaluation functions

def check_length(response: str, min_length: int = 10, max_length: int = 500) -> bool:
    """Check if response length is within bounds"""
    return min_length <= len(response) <= max_length

def check_keywords(response: str, keywords: list) -> bool:
    """Check if response contains required keywords"""
    response_lower = response.lower()
    return all(keyword.lower() in response_lower for keyword in keywords)

def check_no_profanity(response: str) -> bool:
    """Simple check for inappropriate content"""
    # This is a simplified example
    inappropriate = ["bad_word"]  # Replace with actual list
    response_lower = response.lower()
    return not any(word in response_lower for word in inappropriate)

print("=== Testing LLM Responses ===")
# Test LLM outputs

prompt = ChatPromptTemplate.from_messages([
    ("human", "Explain {topic} in simple terms."),
])

chain = prompt | llm | StrOutputParser()

test_cases = [
    {"topic": "Python", "expected_keywords": ["programming", "language"]},
    {"topic": "AI", "expected_keywords": ["artificial", "intelligence"]},
]

print("Testing LLM responses:")
for test in test_cases:
    response = chain.invoke({"topic": test["topic"]})
    
    print(f"\nTopic: {test['topic']}")
    print(f"Response: {response[:100]}...")
    
    # Evaluate
    length_ok = check_length(response)
    keywords_ok = check_keywords(response, test["expected_keywords"])
    
    print(f"  Length check: {'✓' if length_ok else '✗'}")
    print(f"  Keywords check: {'✓' if keywords_ok else '✗'}")
    
    if length_ok and keywords_ok:
        print("  Overall: ✓ PASS")
    else:
        print("  Overall: ✗ FAIL")

print()

print("=== Comparison Evaluation ===")
# Compare different configurations

def evaluate_response(response: str, criteria: dict) -> dict:
    """Evaluate response against criteria"""
    results = {}
    
    if "min_length" in criteria:
        results["length"] = len(response) >= criteria["min_length"]
    
    if "required_words" in criteria:
        response_lower = response.lower()
        results["keywords"] = all(
            word.lower() in response_lower 
            for word in criteria["required_words"]
        )
    
    if "max_length" in criteria:
        results["not_too_long"] = len(response) <= criteria["max_length"]
    
    return results

# Test with different temperatures
prompt = ChatPromptTemplate.from_messages([
    ("human", "Write a creative tagline about {topic}."),
])

temperatures = [0.1, 0.7, 0.9]
topic = "technology"

print(f"Comparing different temperatures for topic: {topic}")
for temp in temperatures:
    test_llm = AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        temperature=temp,
    )
    test_chain = prompt | test_llm | StrOutputParser()
    response = test_chain.invoke({"topic": topic})
    
    criteria = {
        "min_length": 10,
        "max_length": 200,
        "required_words": ["technology"]
    }
    
    evaluation = evaluate_response(response, criteria)
    score = sum(evaluation.values()) / len(evaluation) if evaluation else 0
    
    print(f"\nTemperature: {temp}")
    print(f"  Response: {response[:80]}...")
    print(f"  Evaluation: {evaluation}")
    print(f"  Score: {score:.2%}")

print()

print("=== Semantic Similarity Evaluation ===")
# Evaluate semantic similarity (conceptual)

def semantic_similarity_score(response: str, expected_concepts: list) -> float:
    """Calculate semantic similarity score (simplified)"""
    # In practice, you'd use embeddings for this
    response_lower = response.lower()
    matches = sum(1 for concept in expected_concepts if concept.lower() in response_lower)
    return matches / len(expected_concepts) if expected_concepts else 0.0

test_query = "What is machine learning?"
expected_concepts = ["learning", "data", "algorithm", "pattern"]

response = llm.invoke(test_query)
similarity = semantic_similarity_score(response.content, expected_concepts)

print(f"Query: {test_query}")
print(f"Response: {response.content[:100]}...")
print(f"Semantic similarity score: {similarity:.2%}")
print(f"Expected concepts: {expected_concepts}\n")

print("=== Evaluation Metrics ===")
print("""
Common metrics:
1. Accuracy: Correctness of answers
2. Relevance: How relevant is the response
3. Completeness: Does it cover all aspects
4. Length: Appropriate length
5. Coherence: Does it make sense
6. Factuality: Are claims accurate
7. Toxicity: Is content appropriate

Advanced metrics:
- BLEU score (for translation)
- ROUGE score (for summarization)
- Embedding similarity
- Human evaluation
""")

print("=== Evaluation Framework ===")
# Simple evaluation framework

class LLMEvaluator:
    """Simple LLM evaluator"""
    
    def __init__(self):
        self.test_results = []
    
    def evaluate(self, response: str, criteria: dict) -> dict:
        """Evaluate a response"""
        result = {
            "response": response,
            "criteria": criteria,
            "scores": {},
            "overall": 0.0,
        }
        
        # Length check
        if "length_range" in criteria:
            min_len, max_len = criteria["length_range"]
            result["scores"]["length"] = min_len <= len(response) <= max_len
        
        # Keyword check
        if "keywords" in criteria:
            response_lower = response.lower()
            matches = sum(1 for kw in criteria["keywords"] if kw.lower() in response_lower)
            result["scores"]["keywords"] = matches / len(criteria["keywords"])
        
        # Calculate overall score
        if result["scores"]:
            result["overall"] = sum(result["scores"].values()) / len(result["scores"])
        
        self.test_results.append(result)
        return result
    
    def get_summary(self) -> dict:
        """Get evaluation summary"""
        if not self.test_results:
            return {}
        
        avg_score = sum(r["overall"] for r in self.test_results) / len(self.test_results)
        return {
            "total_tests": len(self.test_results),
            "average_score": avg_score,
            "passed": sum(1 for r in self.test_results if r["overall"] > 0.7),
        }

# Use evaluator
evaluator = LLMEvaluator()

test_responses = [
    "Python is a programming language used for data science and web development.",
    "AI stands for artificial intelligence and machine learning.",
]

criteria = {
    "length_range": (20, 200),
    "keywords": ["language", "programming", "Python"]  # For first response
}

print("Running evaluations:")
for i, response in enumerate(test_responses):
    test_criteria = criteria.copy()
    if i == 1:
        test_criteria["keywords"] = ["artificial", "intelligence", "AI"]
    
    result = evaluator.evaluate(response, test_criteria)
    print(f"\nTest {i+1}:")
    print(f"  Response: {response}")
    print(f"  Scores: {result['scores']}")
    print(f"  Overall: {result['overall']:.2%}")

summary = evaluator.get_summary()
print(f"\nSummary: {summary}\n")

print("=== Best Practices ===")
print("""
1. Define clear criteria
   - What makes a good response?
   - Set measurable metrics
   - Consider use case

2. Test systematically
   - Create test cases
   - Test edge cases
   - Test with different inputs

3. Monitor over time
   - Track metrics
   - Compare versions
   - Identify regressions

4. Combine methods
   - Automated checks
   - Human evaluation
   - Embedding similarity

5. Iterate
   - Use results to improve
   - Refine criteria
   - Update test cases
""")

print("Tutorial complete! You've learned how to evaluate LLM outputs.")

