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
**Code**: https://github.com/lakshmanok/generative-ai-design-patterns/tree/main/examples/01_logits_masking   
**Workarounds (for API models or to replace Logits Masking)**: 
  - 1. stop words.
  - 2. post-generation validation with retry.
  - 3. [grammar rule](#2-grammar). If the rules you want to apply can be represented in certain types of
standard forms, you can offload Logits Masking to the model
provider by providing it with the rules you want to impose.
  - 4. few-shot examples in the context + [Style Transfer](3-style-transfer) and
providing detailed instructions in the prompt through prompt
engineering. Useful for poetry. However, these do not provide a strict enforcement
mechanism—you can’t be sure that your generated text will
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

---
### 3. Style Transfer

---
### 4. Reverse Neutralization

---
### 5. Content Optimization
