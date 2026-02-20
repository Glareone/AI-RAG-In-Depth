# LangGraph Document Analysis Prototype

A modular document analysis application built with LangGraph that uses the ReAct pattern for intelligent rule extraction and validation.

## Features

- Document processing with intelligent chunking
- Rule extraction using keyword matching and NLP
- ReAct pattern for LLM-based rule validation
- Iterative confidence assessment and enrichment
- Conditional workflow routing based on confidence scores
- Azure OpenAI integration with reasoning models (gpt-5, gpt-5-mini, gpt-5-nano)

## Project Structure

```
LangGraph_Prototype/
├── main.py              # Clean entry point - orchestrates the workflow
├── config.py            # Configuration, constants, and LLM initialization
├── memory.py            # State schema and memory management
├── tools.py             # Utility functions and rule loading from JSON
├── nodes.py             # Graph node functions (processing steps)
├── graph.py             # Graph construction and compilation
├── __init__.py          # Package initialization
├── data/                # Data directory
│   ├── example.txt      # Sample document for testing
│   └── sample_rules.json # Rule definitions (required)
└── requirements.txt     # Python dependencies
```

## Module Descriptions

- **main.py**: Entry point that runs the document analysis workflow
- **config.py**: Contains LLM configuration and application constants
- **memory.py**: Defines `DocumentState` schema and state initialization
- **tools.py**: File I/O utilities and rule loading from `data/sample_rules.json`
- **nodes.py**: Individual processing nodes (chunk, extract, validate, assess, enrich)
- **graph.py**: LangGraph workflow construction and compilation
- **data/**: Contains sample documents and rule definitions (JSON format)

## Setup

1. Install dependencies using uv:
```bash
uv pip install -r requirements.txt
```

2. Create a `.env` file in the root directory with your Azure OpenAI credentials:
```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

3. Configure your model in `config.py`:
   - **gpt-5**: Larger model, best for complex tasks (slower, more expensive)
   - **gpt-5-mini**: Balanced performance (default)
   - **gpt-5-nano**: Fastest and cheapest option

4. Place your text files in the `data` directory.

## Running the Application

```bash
python main.py
```

## Usage as a Module

The modular structure makes it easy to import and use in other applications:

```python
from memory import create_initial_state
from tools import read_file
from graph import app

# Read your document
content = read_file("path/to/document.txt")

# Create initial state
initial_state = create_initial_state(content)

# Run the workflow
result = app.invoke(initial_state)

# Access results
print(f"Validated rules: {len(result['validated_rules'])}")
print(f"Confidence: {result['confidence_score']}")
```

## FastAPI Integration

This structure is designed to be easily integrated into a FastAPI application:
- Import the `app` from `graph.py` to process documents
- Use `create_initial_state()` to initialize state from request data
- Return results as JSON responses 