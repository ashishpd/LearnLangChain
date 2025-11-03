# LearnLangChain

A comprehensive tutorial series for learning LangChain from basics to advanced concepts. This repository contains 30 progressive tutorials with working code examples, all using Azure OpenAI as the LLM provider.

## Overview

This tutorial series is designed to take you from beginner to advanced LangChain practitioner through hands-on examples. Each tutorial builds upon previous concepts, with clear explanations and practical code examples.

## Prerequisites

- Python 3.8 or higher
- Azure OpenAI account with a deployment
- Basic Python programming knowledge
- Familiarity with command line

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd LearnLangChain
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Azure OpenAI

Create a `.env` file in the root directory with your Azure OpenAI credentials:

```env
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Optional: If you have a separate embedding deployment
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your_embedding_deployment_name
```

### 4. Run Your First Tutorial

```bash
python 01_hello_world.py
```

## Tutorial Structure

### Part 1: Foundations (01-07)
Learn the basics of LangChain and Azure OpenAI integration.

- **01_hello_world.py** - Introduction and basic LLM invocation
- **02_llm_providers.py** - Azure OpenAI configuration options
- **03_prompt_templates.py** - Creating and using prompt templates
- **04_output_parsers.py** - Parsing and structuring LLM outputs
- **05_chains_intro.py** - Introduction to chains
- **06_sequential_chains.py** - Building sequential workflows
- **07_router_chains.py** - Conditional routing and multi-route processing

### Part 2: Memory & State Management (08-09)
Learn how to maintain conversation state.

- **08_memory_basic.py** - Basic memory and conversation context
- **09_memory_types.py** - Different memory types and when to use them

### Part 3: Agents (10-13)
Build intelligent agents that can use tools and make decisions.

- **10_agents_intro.py** - Introduction to agents
- **11_custom_tools.py** - Creating custom tools for agents
- **12_multi_tool_agents.py** - Agents with multiple tools
- **13_agent_memory.py** - Combining agents with memory

### Part 4: Document Processing & RAG (14-19)
Learn to work with documents and build RAG systems.

- **14_document_loaders.py** - Loading documents from various sources
- **15_text_splitters.py** - Splitting text into chunks
- **16_embeddings.py** - Understanding and using embeddings
- **17_vector_stores.py** - Storing and searching vectors
- **18_rag_basic.py** - Basic RAG implementation
- **19_rag_advanced.py** - Advanced RAG patterns and techniques

### Part 5: Advanced Topics (20-27)
Master advanced LangChain features and production patterns.

- **20_streaming.py** - Streaming responses for better UX
- **21_callbacks.py** - Monitoring and debugging with callbacks
- **22_chains_advanced.py** - Advanced chain architectures
- **23_custom_chains.py** - Building custom chain classes
- **24_async_operations.py** - Async/await patterns and concurrency
- **25_evaluation.py** - Evaluating LLM outputs
- **26_debugging.py** - Debugging strategies and tips
- **27_production_patterns.py** - Production best practices

### Part 6: Integration Examples (28-30)
Real-world integration examples.

- **28_web_scraper.py** - Web scraping with LangChain
- **29_sql_chain.py** - Database queries and SQL integration
- **30_complete_app.py** - Complete end-to-end application

## Learning Path

### Beginner Path (Start Here)
1. Start with **01_hello_world.py** to set up your environment
2. Progress through Part 1 (01-07) to learn fundamentals
3. Learn about memory (08-09) for conversational applications
4. Total: ~2-3 hours

### Intermediate Path
1. Complete beginner path first
2. Learn agents (10-13) for dynamic workflows
3. Master document processing (14-19) for RAG applications
4. Total: ~5-6 hours

### Advanced Path
1. Complete intermediate path
2. Study advanced topics (20-27) for production use
3. Review integration examples (28-30) for real-world patterns
4. Total: ~8-10 hours

## Key Features

- **Progressive Learning**: Each tutorial builds on previous concepts
- **Working Examples**: All code is runnable and tested
- **Azure OpenAI Focus**: Specifically configured for Azure OpenAI
- **Well Commented**: Extensive comments explain concepts
- **Production Ready**: Includes best practices and patterns

## Common Issues

### Issue: Module not found errors
**Solution**: Make sure you've installed all dependencies:
```bash
pip install -r requirements.txt
```

### Issue: Azure OpenAI authentication errors
**Solution**: 
1. Check your `.env` file has correct credentials
2. Verify your Azure OpenAI deployment is active
3. Ensure API key has proper permissions

### Issue: Import errors
**Solution**: Make sure you're using the correct LangChain version. Some imports may vary by version.

## Project Structure

```
LearnLangChain/
├── README.md                 # This file
├── TUTORIAL_PLAN.md          # Detailed tutorial plan
├── requirements.txt          # Python dependencies
├── .env.example              # Example environment variables
├── 01_hello_world.py         # Tutorial files
├── 02_llm_providers.py
├── ...
└── 30_complete_app.py
```

## Contributing

This is a learning repository. Feel free to:
- Report issues
- Suggest improvements
- Add more examples
- Improve documentation

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Azure OpenAI Documentation](https://learn.microsoft.com/azure/cognitive-services/openai/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)

## License

This tutorial series is provided for educational purposes.

## Next Steps

After completing the tutorials:

1. **Build Your Own Project**: Apply what you've learned to a real project
2. **Explore LangChain Ecosystem**: Check out LangChain integrations
3. **Join Community**: Engage with LangChain community
4. **Experiment**: Try different models, configurations, and patterns

## Support

If you encounter issues:
1. Check the tutorial comments for guidance
2. Review the [LangChain documentation](https://python.langchain.com/)
3. Check your Azure OpenAI setup

---

Happy Learning! 🚀
