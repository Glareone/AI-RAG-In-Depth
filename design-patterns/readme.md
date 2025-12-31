## GenAI Design Patterns

---
### 1. Logits Masking

**Problem**: Generate which adheres the text set of rules. Regeneration isnt working all the time, but also increases the latency and costs.  

```txt
The naive approach, or antipattern, is to generate content, evaluate it
against the relevant rules, and regenerate it if the content doesn’t conform to
the rules (see Figure 2-1). However, such a “try-and-try-again” approach
only works for edge cases when very few responses (less than 10% or so)
need to be regenerated. Otherwise, multiple retries will dramatically
increase latency and sometimes won’t even converge toward an acceptable
answer.
```
<img width="893" height="585" alt="image" src="https://github.com/user-attachments/assets/50f005c2-96ef-48ac-8315-a4eb5931deef" />

```txt
The average number of generations required before you get a
successful one is given by this formula: 100/p. For example, if
10% of your generations fail, then 90% succeed—so you will
need 100/90, or 1.1, attempts on average, and the average
latency goes up by 10%. On the other hand, if 70% of your
generations fail, only 30% succeed—so you will need 100/30,
or 3.3, attempts on average
```

**Solution**: The idea behind LogitsMasking is to intercept the generation at this sampling stage.  
```
Logits Masking (low-level):
User Input → Tokenizer → Model → [INTERCEPT LOGITS] → Sample → Output
                                        ↑
                                  You modify here

LangGraph (high-level):
User Input → LangGraph Node → LLM API → Output
                                ↑
                          Black box - no logits access
```


```txt
1. Rather than wait until the full content is generated, you obtain the
set of possible continuations at each intermediate point.
You zero out the probability associated with continuations that do
not meet the rules.

2. As long as there is at least one continuation that meets the rules,
generation can proceed.

3. If there is no continuation that meets the rules or if the generation
is at a point that you have previously encountered as a dead end,
you need to back up one step and retry generation.

4. After some maximum number of generation attempts, you send a
refusal back to the user saying that you are unable to generate
content that meets the rules.
```

<img width="706" height="980" alt="image" src="https://github.com/user-attachments/assets/7eae53ff-d77b-498a-836e-f706f0166482" />

**Limitations**:
```
But this only works with models where logits :

✅ Local/self-hosted models (Llama, Mistral, etc.)
✅ HuggingFace Transformers
❌ NOT with API-based models (OpenAI, Anthropic, Cohere)


```

| Requirement | Open-Source (Llama) | API Models (Claude/GPT-4) | 
| ----------- | ------------------- | ------------------------- |
| Access to weights | ✅ Yes | ❌ No (proprietary) |
| Logits exposure   | ✅ Yes (HF Transformers) | ❌ Hidden | 
| Custom sampling   | ✅ Yes | ❌ Standardized API |

**Code**: https://github.com/lakshmanok/generative-ai-design-patterns/tree/main/examples/01_logits_masking  
**Workaround for API Models**: stop words, post-generation validation with retry, [grammar rule](#2-grammar)

---
### 2. Grammar

---
### 3. Style Transfer

---
### 4. Reverse Neutralization

---
### 5. Content Optimization
