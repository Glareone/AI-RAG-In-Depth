## GenAI Design Patterns

---
### [1. Logits Masking. Click to See detailed information](https://github.com/Glareone/AI-RAG-In-Depth/blob/main/design-patterns/logits_masking.md)
```
When to Use Logits Masking Pattern:
✅ 1. Open-source models (Llama, Mistral, etc.)
✅ 2. Self-hosted deployments
✅ 3. HuggingFace Transformers
✅ 4. Maximum control requirements
❌ 5. API-based models (Claude, GPT-4)
❌ 6. LangGraph with external APIs
❌ 7. When simpler alternatives suffice
```

**Problem**: Generate which adheres the text set of rules. Regeneration isnt working all the time, but also increases the latency and costs.  
**Solution**: The idea behind LogitsMasking is to intercept the generation at this sampling stage.   
**Limitations**: This only works with models where logits is available and the latency is not critical.  

**Code**:
1) [Structured Output, outlines, NER for Named Entity Extraction (N-Graph)](https://github.com/Glareone/AI-RAG-In-Depth/blob/main/design-patterns/1_constrained_decoding_openai.ipynb)  
2) [True logits masking](https://github.com/Glareone/AI-RAG-In-Depth/blob/main/design-patterns/2_true_logits_masking_llama.ipynb)    

**Workarounds (for API models or to replace Logits Masking)**: 
  - 1. stop words.
  - 2. post-generation validation with retry.
  - 3. [grammar rule](#2-grammar). If the rules you want to apply can be represented in certain types of
standard forms, you can offload Logits Masking to the model
provider by providing it with the rules you want to impose.
  - 4. few-shot examples in the context + [Style Transfer](3-style-transfer) and
providing detailed instructions in the prompt through prompt
engineering. Useful for poetry. However, these do not provide a strict enforcement
mechanism - you can’t be sure that your generated text will
conform to the rules.
  - 5. Using a more powerful model might be an option because such
models are typically better at following instructions.

**Considerations**:
```
Logits Masking is a way of using much of the machinery of the LLM to
generate text while imposing your preferences on the sampling. It’s useful
when continuation sequences can easily be censored to remove disallowed
options. The simple sequence selection approach works when censoring tends to
leave behind a few valid options. In more complex scenarios, where it is
highly likely that censoring will remove all of the generated options, you
might need to backtrack and regenerate sequences.
```

---
### 2. [Grammar](https://github.com/Glareone/AI-RAG-In-Depth/blob/main/design-patterns/grammar.md)

The Grammar pattern provides a way to ensure that generated text adheres style rules that can be represented as a context-free `metasyntax`.  


<img width="450" height="800" alt="image" src="https://github.com/user-attachments/assets/4e49c734-c87c-41af-91a8-9caca4699b2e" />

### Problem
Generate the text which follows the specific format:   
1) Simple as a comma-separated list  
2) JSON  
3) Complex syntactically valid structured query language (SQL)  
4) Math expression
5) Pipe-separated information for UNIONS or for tables in .Markdown

### Solution
1. Either prepare the `grammar` and inject it into the `logits_processor`  
2. Use predefined `grammar-constrained logits_processor`  
3. Or use more user-friendly options, such as specifying a data format or passing in a schema description    

#### How to
1. Represent the syntax you want in the form of a formal grammar (In Progress)
2. Create a LogitsProcessor with applied grammar. (In Progress)
3. Pass in the logits processor to the pipeline. (In Progress)


**Problem**: LLMs may generate output in unpredictable formats or include unwanted explanations, making it difficult to parse and process responses programmatically.

**Solution**: Use grammar constraints to physically block invalid tokens at generation time, ensuring 100% format compliance.

**Notebook**: [2_grammar_pattern_4_examples.ipynb](https://github.com/Glareone/AI-RAG-In-Depth/blob/main/design-patterns/2_grammar_pattern_4_examples.ipynb)

**Examples**:
1. **Insurance Forms** - Complex nested JSON extraction using Pydantic schemas
2. **SQL Query Generation** - Generate safe SQL with TRUE BNF Grammar Pattern
3. **Pipe-Separated Data** - Extract structured data with strict format
4. **English Grammar Correction** - Fix grammar while constraining output format
5. **🎓 Math Expression Generation** - Deep dive into direct logits processing

**Key Learning**: Grammar Pattern provides 100% guarantee of valid output format through token-level logits masking.

#### Approaches Comparison

| Feature | Grammar Pattern (outlines/llama-cpp/IncrementalGrammarConstraint) | Structured Outputs (Azure OpenAI) |
|---------|---------------------------------------|-----------------------------------|
| Implementation | Self-hosted model                     | Azure OpenAI API |
| Constraint Type | Token-level grammar                   | Schema + parsing |
| Safety Guarantee | ✅ HARD (impossible to violate)        | ⚠️ SOFT (99% reliable) |
| Grammar Support | ✅ BNF, regex, FSM                     | ❌ Not supported |
| Output Structure | ⚠️ Text (must match grammar)          | ✅ Pydantic objects |
| Model Control | ✅ Full control                        | ❌ Server-side only |

#### 🎓 Deep Dive: How Grammar Constraints Work Internally

**Example 5** in the notebook demonstrates the **foundational mechanism** that libraries like `outlines` and `llama-cpp` use under the hood:

**Direct Logits Processing**:
1. Grammar is compiled into a finite state machine
2. At each token generation step:
   - Determine which tokens are valid per current grammar state
   - Set logits of INVALID tokens to `-inf` (impossible to select)
   - Model MUST choose from valid tokens
3. Result: 100% guaranteed format compliance

**Abstraction Levels**:
- **High Level**: Pydantic schemas with OpenAI API (Examples 1, 3)
- **Medium Level**: `outlines` library with regex/BNF (Examples 2, 4)
- **Low Level**: Direct `GrammarConstrainedLogitsProcessor` (Example 5) ← **Shows how it actually works**

**Libraries**:
- **outlines**: HuggingFace models, auto-downloads, regex/JSON/FSM support
- **llama-cpp**: GGUF models (quantized), manual download, full BNF grammar
- **transformers-cfg**: Direct logits processing for educational purposes

**When to Use**:
```
✅ Generate valid SQL queries (block DELETE/UPDATE/DROP)
✅ Extract structured data with guaranteed format
✅ Math expression generation (no explanations)
✅ Fix grammar while constraining output structure
✅ Any scenario requiring 100% format compliance
❌ When simpler prompt engineering suffices
❌ API-based models without grammar support
```

**Code Reference**: See [Example 5](https://github.com/Glareone/AI-RAG-In-Depth/blob/main/design-patterns/2_grammar_pattern_4_examples.ipynb) for direct logits processing implementation.

---
### 3. Style Transfer

---
### 4. Reverse Neutralization

---
### 5. Content Optimization
