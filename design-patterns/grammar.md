# Pattern 2: Grammar

## Overview

**Problem**: LLMs may generate output in unpredictable formats, include unwanted explanations, or produce invalid syntax, making responses difficult to parse and process programmatically.

**Solution**: Use grammar constraints to physically block invalid tokens at generation time through **logits masking**, ensuring 100% format compliance.

**Key Guarantee**: Unlike prompt engineering (soft constraint), grammar pattern provides **hard constraint** - it's physically impossible for the model to generate tokens outside the defined grammar.

---

## 📚 Notebook: 5 Battle-Tested Examples

**[2_grammar_pattern_4_examples.ipynb](https://github.com/Glareone/AI-RAG-In-Depth/blob/main/design-patterns/2_grammar_pattern_4_examples.ipynb)**

### Educational Progression

The notebook teaches Grammar Pattern through **abstraction levels**:

| Level | Approach | Examples | What You Learn |
|-------|----------|----------|----------------|
| **High** | Pydantic + Azure OpenAI | 1, 3 | Easiest to use - define schema, get results |
| **Medium** | outlines/llama-cpp | 2, 4 | Library handles logits - you define grammar |
| **Low** | Direct logits processing | 5 | **How it actually works** under the hood |

### The 5 Examples

#### Example 1: Insurance Forms
- **Use Case**: Extract complex nested JSON from unstructured claim forms
- **Approach**: Pydantic schemas with Azure OpenAI Structured Outputs
- **What It Shows**: High-level API approach, easiest to implement
- **Key Learning**: Define schema once, get guaranteed structure

#### Example 2: SQL Query Generation
- **Use Case**: Generate safe SQL queries, block dangerous operations (DELETE, UPDATE, DROP)
- **Approach**: BNF Grammar with `outlines` (regex) and `llama-cpp` (full BNF)
- **What It Shows**: True grammar constraints, safety guarantees
- **Key Learning**: Grammar physically blocks dangerous tokens - DELETE/UPDATE become impossible

#### Example 3: Pipe-Separated Data
- **Use Case**: Extract product information in legacy system format (SKU|Name|Price|Category)
- **Approach**: Pydantic schemas converted to pipe-separated format
- **What It Shows**: Structured extraction for legacy system integration
- **Key Learning**: Guaranteed format for downstream systems

#### Example 4: English Grammar Correction
- **Use Case**: Fix grammar errors, output ONLY corrected sentence (no explanations)
- **Approach**: Regex grammar with `outlines`
- **What It Shows**: Constraining natural language output format
- **Key Learning**: Block verbose explanations, guarantee concise output

#### Example 5: 🎓 Math Expression Generation (Deep Dive)
- **Use Case**: Extract mathematical expressions from word problems
- **Approach**: **Direct logits processing** with `GrammarConstrainedLogitsProcessor`
- **What It Shows**: **THE FOUNDATIONAL MECHANISM** - how grammar constraints actually work
- **Key Learning**: See token-level logits masking in action - understand WHY it's 100% guaranteed

---

## How Grammar Constraints Work (Example 5 Reveals This)

### The Mechanism

1. **Grammar Compilation**: BNF/regex grammar → finite state machine (FSM)
2. **Token Generation Loop**: At each step during generation:
   - FSM determines which tokens are valid in current state
   - **Logits of INVALID tokens are set to `-inf`** (impossible to select)
   - Model MUST choose from valid tokens only
3. **Result**: Output guaranteed to match grammar - violations are physically impossible

### What Libraries Abstract Away

```python
# High-level (outlines)
generator = Generator(model, output_type=regex("SELECT.*"))
result = generator(prompt)  # ✅ Magic happens inside

# What's happening under the hood (Example 5 shows this)
grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
processor = GrammarConstrainedLogitsProcessor(grammar)  # <-- Sets logits to -inf!
result = model.generate(prompt, logits_processor=[processor])
```

**Example 5 is crucial** because it shows this mechanism explicitly, helping you understand:
- Why grammar constraints provide 100% guarantees
- The difference between soft (prompt) and hard (logits) constraints
- What `outlines` and `llama-cpp` do internally

---

## Approaches Comparison

### Structured Outputs vs Grammar Pattern

| Feature | Grammar Pattern | Structured Outputs |
|---------|----------------|-------------------|
| | (outlines/llama-cpp) | (Azure OpenAI) |
| Implementation | Self-hosted model | Azure OpenAI API |
| Constraint Type | Token-level grammar | Schema + parsing |
| **Safety Guarantee** | ✅ **HARD (impossible)** | ⚠️ **SOFT (99% reliable)** |
| Grammar Support | ✅ BNF, regex, FSM | ❌ Not supported |
| SQL Safety | ✅ DELETE physically blocked | ⚠️ Relies on prompt |
| Output Structure | ⚠️ Text (must match grammar) | ✅ Pydantic objects |
| Setup | Medium (install libraries) | Low (just API) |
| Cost | Hardware only | Per-token pricing |
| Control | ✅ Full control | ❌ Server-side only |

### Library Comparison

| Feature | outlines | llama-cpp | transformers-cfg |
|---------|----------|-----------|------------------|
| **Model Format** | HuggingFace | GGUF | HuggingFace |
| **Precision** | FP16/BF16 | Quantized (Q4-Q8) | FP16/BF16 |
| **Download** | ✅ Auto | ⚠️ Manual | ✅ Auto |
| **Grammar** | Regex, JSON, FSM | ✅ Full BNF | BNF |
| **Model Size** | ~6GB | ~2.5GB (Q5) | ~6GB |
| **Best For** | Latest models, GPU | CPU, smaller footprint | **Educational** |
| **Examples** | 2, 4 | 2 (alternative) | **5 (shows mechanism)** |

---

## When to Use Grammar Pattern

### ✅ Use Grammar Pattern When:

1. **100% Format Compliance Required**
   - Database queries (block dangerous operations)
   - Legacy system integration (exact format needed)
   - Mathematical expressions (no natural language)

2. **Safety-Critical Scenarios**
   - SQL generation (block DELETE, UPDATE, DROP)
   - Code generation (prevent injection attacks)
   - Command generation (restrict to safe operations)

3. **Self-Hosted Models**
   - You control the model deployment
   - Have access to logits/token probabilities
   - Using HuggingFace transformers or llama-cpp

4. **Complex Grammar Rules**
   - BNF grammars for programming languages
   - Regular expressions for structured formats
   - FSM for state-based generation

### ❌ Don't Use When:

1. **API-Based Models** (Claude, GPT-4)
   - No access to logits
   - Use Structured Outputs instead (softer guarantee)

2. **Simple Prompting Suffices**
   - Model reliably follows instructions
   - Minor format deviations are acceptable

3. **No Grammar Definition Possible**
   - Output format is too flexible/creative
   - Natural language generation without constraints

---

## Code Examples

### Example: Safe SQL Generation (Example 2)

```python
from outlines import Generator, regex

# Define regex that ONLY allows SELECT
sql_regex = r"SELECT[\s\S]+FROM[\s\S]+"
sql_generator = Generator(model, output_type=regex(sql_regex))

# Try to get dangerous operation
prompt = "Delete all employees from Engineering"
result = sql_generator(prompt)
# Output: SELECT * FROM employees WHERE department = 'Engineering'
# ✅ Grammar forced SELECT even when asked for DELETE!
```

### Example: Math Expressions (Example 5 - The Deep Dive)

```python
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

# Define BNF grammar
grammar_str = """
root ::= (expr "=" ws term "\\n")+
expr ::= term ([-+*/] term)*
term ::= ident | num | "(" ws expr ")" ws
ident ::= [a-z] [a-z0-9_]* ws
num ::= [0-9]+ ws
ws ::= [ \\t\\n]*
"""

# Create logits processor
grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
processor = GrammarConstrainedLogitsProcessor(grammar)

# Generate with constraint
prompt = "Bill has 3 apples, Mae has 2. How many total?"
result = model.generate(prompt, logits_processor=[processor])
# Output: bill_apples + mae_apples = 5
# ✅ Only math expression - explanations are IMPOSSIBLE!
```

---

## Key Takeaways

1. **Grammar Pattern = Hard Constraint**
   - Physically blocks invalid tokens (logits = -inf)
   - 100% guarantee of format compliance
   - Different from prompt engineering (soft constraint)

2. **Abstraction Levels Matter**
   - High: Pydantic/API - easiest to use
   - Medium: outlines/llama-cpp - library handles logits
   - Low: Direct processing - understand the mechanism

3. **Example 5 is Educational Gold**
   - Shows HOW grammar constraints work
   - Reveals the logits masking mechanism
   - Makes you appreciate library abstractions

4. **Choose Based on Use Case**
   - Safety-critical → Grammar Pattern (hard guarantee)
   - Convenience → Structured Outputs (API)
   - Understanding → Example 5 (learn the mechanism)

---

## Resources

- **Notebook**: [2_grammar_pattern_4_examples.ipynb](https://github.com/Glareone/AI-RAG-In-Depth/blob/main/design-patterns/2_grammar_pattern_4_examples.ipynb)
- **Libraries**:
  - [outlines](https://github.com/outlines-dev/outlines) - HuggingFace integration
  - [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - GGUF models
  - [transformers-cfg](https://github.com/epfl-dlab/transformers-CFG) - Direct logits (educational)
- **Book Reference**: "Generative AI Design Patterns" - Grammar Pattern chapter