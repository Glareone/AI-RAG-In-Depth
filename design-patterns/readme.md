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
### 2. Grammar

The Grammar pattern provides a way to ensure that generated text adheres style rules that can be represented as a context-free `metasyntax`.  

<img width="775" height="683" alt="image" src="https://github.com/user-attachments/assets/e5485356-b283-42e7-8e52-229a60252681" />

### Problem
Generate the text which follows the specific format:   
1) Simple as a comma-separated list  
2) JSON  
3) Complex syntactically valid structured query language (SQL)  
4) Math expression  

### Solution
1. Either prepare the `grammar` and inject it into the `logits_processor`  
2. Use predefined `grammar-constrained logits_processor`  
3. Or use more user-friendly options, such as specifying a data format or passing in a schema description    

#### How to
1. Represent the syntax you want in the form of a formal grammar (In Progress)
2. Create a LogitsProcessor with applied grammar. (In Progress)
3. Pass in the logits processor to the pipeline. (In Progress)


---
### 3. Style Transfer

---
### 4. Reverse Neutralization

---
### 5. Content Optimization
