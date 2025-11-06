"""
27_production_patterns.py - Production Best Practices

This tutorial demonstrates:
- Production best practices
- Error handling
- Configuration management
- Security considerations
- Monitoring and logging

Prerequisites:
- Understanding of LangChain concepts
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Optional
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=== Production Best Practices ===")
print("""
Key considerations:
1. Error handling
2. Configuration management
3. Security
4. Logging and monitoring
5. Performance optimization
6. Testing and validation
""")

print("=== Configuration Management ===")
# Centralized configuration

class Config:
    """Centralized configuration"""
    
    def __init__(self):
        self.azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.temperature = float(os.environ.get("TEMPERATURE", "0.7"))
        self.max_retries = int(os.environ.get("MAX_RETRIES", "3"))
        self.timeout = int(os.environ.get("TIMEOUT", "30"))
        
        # Validate required settings
        if not self.azure_deployment:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT_NAME not set")
    
    def get_llm_config(self):
        """Get LLM configuration"""
        return {
            "azure_deployment": self.azure_deployment,
            "temperature": self.temperature,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }

config = Config()
print(f"Configuration loaded:")
print(f"  Deployment: {config.azure_deployment}")
print(f"  Temperature: {config.temperature}")
print(f"  Max retries: {config.max_retries}")
print()

print("=== Error Handling ===")
# Robust error handling

class ProductionLLMChain:
    """Production-ready chain with error handling"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = AzureChatOpenAI(**config.get_llm_config())
        self.chain = None
    
    def initialize_chain(self, prompt_template):
        """Initialize chain with error handling"""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("human", prompt_template),
            ])
            self.chain = prompt | self.llm | StrOutputParser()
            logger.info("Chain initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize chain: {e}")
            raise
    
    def run_with_retry(self, inputs: dict, max_retries: int = None) -> Optional[str]:
        """Run chain with retry logic"""
        max_retries = max_retries or self.config.max_retries
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}")
                result = self.chain.invoke(inputs)
                logger.info("Chain execution successful")
                return result
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error("All retry attempts failed")
                    return None
                # Could add exponential backoff here
        
        return None

# Use production chain
prod_chain = ProductionLLMChain(config)
prod_chain.initialize_chain("Explain {topic} briefly.")

print("Testing production chain with error handling:")
result = prod_chain.run_with_retry({"topic": "Python"})
if result:
    print(f"Result: {result}\n")

print("=== Input Validation ===")
# Validate and sanitize inputs

def validate_and_sanitize_input(input_data: dict) -> dict:
    """Validate and sanitize input data"""
    sanitized = {}
    
    for key, value in input_data.items():
        # Type validation
        if not isinstance(value, str):
            raise ValueError(f"Input {key} must be a string")
        
        # Length validation
        if len(value) > 10000:
            raise ValueError(f"Input {key} too long (max 10000 chars)")
        
        # Basic sanitization (strip whitespace)
        sanitized[key] = value.strip()
    
    return sanitized

print("Input validation example:")
try:
    validated = validate_and_sanitize_input({"topic": "  Python  "})
    print(f"Validated input: {validated}")
except ValueError as e:
    print(f"Validation error: {e}")
print()

print("=== Security Best Practices ===")
print("""
Security considerations:

1. API Keys:
   - Never hardcode credentials
   - Use environment variables
   - Rotate keys regularly
   - Use secret management services

2. Input Sanitization:
   - Validate all inputs
   - Sanitize user-provided data
   - Prevent injection attacks
   - Limit input sizes

3. Output Filtering:
   - Review outputs before displaying
   - Filter sensitive information
   - Handle errors without exposing internals

4. Rate Limiting:
   - Implement rate limits
   - Monitor usage
   - Prevent abuse

5. Access Control:
   - Authenticate users
   - Authorize actions
   - Audit access
""")

print("=== Logging Best Practices ===")
# Structured logging

class ProductionLogger:
    """Production logging utility"""
    
    @staticmethod
    def log_chain_execution(chain_name: str, inputs: dict, outputs: dict):
        """Log chain execution"""
        logger.info(f"Chain execution: {chain_name}")
        logger.debug(f"  Inputs: {inputs}")
        logger.debug(f"  Outputs: {outputs}")
    
    @staticmethod
    def log_error(operation: str, error: Exception):
        """Log errors"""
        logger.error(f"Error in {operation}: {type(error).__name__}: {error}")
    
    @staticmethod
    def log_performance(operation: str, duration: float):
        """Log performance metrics"""
        logger.info(f"Performance: {operation} took {duration:.2f}s")

print("Logging example:")
ProductionLogger.log_chain_execution(
    "test_chain",
    {"input": "test"},
    {"output": "result"}
)
print()

print("=== Monitoring ===")
# Basic monitoring setup

import time

class PerformanceMonitor:
    """Simple performance monitor"""
    
    def __init__(self):
        self.metrics = []
    
    def measure(self, operation_name: str):
        """Context manager for measuring operations"""
        return self._Timer(operation_name, self)
    
    class _Timer:
        def __init__(self, name: str, monitor):
            self.name = name
            self.monitor = monitor
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time
            self.monitor.metrics.append({
                "operation": self.name,
                "duration": duration,
                "timestamp": time.time(),
            })
            logger.info(f"Operation {self.name} took {duration:.2f}s")

monitor = PerformanceMonitor()

# Use monitor
with monitor.measure("llm_call"):
    result = llm.invoke("Say hello")
    time.sleep(0.1)  # Simulate processing

print(f"Monitored {len(monitor.metrics)} operations")
print()

print("=== Production Code Structure ===")
print("""
Recommended structure:

project/
├── config/
│   └── settings.py       # Configuration
├── chains/
│   └── custom_chains.py  # Chain definitions
├── utils/
│   ├── logging.py        # Logging setup
│   └── validation.py    # Input validation
├── monitoring/
│   └── metrics.py        # Monitoring
└── main.py               # Application entry

Best practices:
- Separate concerns
- Modular design
- Clear dependencies
- Comprehensive testing
""")

print("=== Production Checklist ===")
print("""
Before deploying:

[ ] Error handling implemented
[ ] Input validation in place
[ ] Logging configured
[ ] Monitoring set up
[ ] Security reviewed
[ ] Configuration externalized
[ ] Tests written
[ ] Documentation updated
[ ] Performance tested
[ ] Rate limiting implemented
[ ] Secrets managed securely
[ ] Deployment process defined
""")

print("Tutorial complete! You've learned production best practices for LangChain.")

