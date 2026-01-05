### 1. Logits Masking

Logits Masking is a brilliant idea of how to inforce LLM to adhere the rules.
There are two different ways how to do this:
1) Inject into the sampling transformer in hosted models (for instance, in Llama)
2) Inject into outline and enforce rules on logits_bias leve.

I compare them below:


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

**Solution**: The idea behind LogitsMasking is to intercept the generation at this sampling stage (or as a work around for API models, on logits_bias level).    
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

<img width="1446" height="2766" alt="image" src="https://github.com/user-attachments/assets/63029d06-5e41-4beb-8127-deefb15c3de3" />

**Limitations**:
```
But this only works with models where logits :

✅ Local/self-hosted models (Llama, Mistral, etc.)
✅ HuggingFace Transformers
⚠️ OpenAI Supports `logit_bias` instead as an alternative to Logits masking (which is also a part of Logits Masking Pattern).
❌ NO WRITE with API-based models (OpenAI, Anthropic, Cohere)
⚠️ OpenAI Supports `logit_bias` instead as an alternative to Logits masking
⚠️ OpenAI & Gemini Flash provides logprobs read capabilities
```

### Side-by-Side Comparison
| Feature | True Logits Masking | Constrained Decoding (outlines + API) | 
| ------- | ------------------- | ------------------------------------- |
| Where it works | Self-hosted only | APIs with logit_bias | 
| Access to logits | ✅ Full array | ❌ None (uses logit_bias) | 
| Complex rules | ✅ Any logic | ❌ Only token bans | 
| Context-aware | ✅ Can check previous tokens | ⚠️ Limited | 
| Dynamic rules | ✅ Rules can change during generation | ❌ Static token list | 
| Flexibility | ✅ Unlimited | ⚠️ Limited by API | 
| Example use case | Legal document compliance | JSON schema enforcement |

### Other capabilities available. Analysis

| Requirement | Self-hosted (Llama-3-8B) | Self-hosted Mistral | Claude | GPT | Gemini Flash | Gemini Pro |
| ----------- | ------------------- | ------------------------- | ---   | --- | ------------ | ---------- |
| Access to weights | ✅ Yes | ✅ Yes | ❌ No (proprietary) | ❌ No | ❌ No | ❌ No |
| Logprobs (read, Observe sampling)      | ✅ Yes | ✅ Yes | ❌ No logits/logprobs at all | ✅ Yes `logprobs=True` |  ✅ Yes `"response_logprobs": True`  | ❌ No |
| Logits Masking (write, Modify sampling logic)    |  ✅ Yes, logits_processor=[MyRulesLogitsProcessor()]  |  ✅ Yes | ❌ No | ❌ No | ❌ No | ❌ No |
| logit_bias (Ban/boost tokens)       | ✅ Yes (vLLM) but ❌ No (HFTransformers)   | ✅ Yes  | ❌ No | ✅ Yes | ❌ No | ❌ No |
| Custom sampling (top-K, top-P)   | ✅ Yes | ✅ Yes | ✅ Yes `temperature, top_p, top_k` | ✅ Yes `temperature, top_p` | ✅ Yes `temperature, top_p, top_k` | ✅ Yes  `temperature, top_p, top_k` |
| Frequency/presence penalties     | ✅ Yes `frequency, presence` (vLLM) but ❌ No for (HFTransformers) | ✅ Yes `frequency, presence` | ❌ No | ✅ Yes | ❌ No | ❌ No |

**Code**: https://github.com/lakshmanok/generative-ai-design-patterns/tree/main/examples/01_logits_masking  

**Workarounds (for API models or to replace Logits Masking)**:   
  - 1. logits_bias (for JSON generation)  
  - 1. stop words  
  - 2. post-generation validation with retry  
  - 3. [grammar rule](#2-grammar). If the rules you want to apply can be represented in certain types of
standard forms, you can offload Logits Masking to the model
provider by providing it with the rules you want to impose  
  - 4. few-shot examples in the context + [Style Transfer](3-style-transfer) and
providing detailed instructions in the prompt through prompt
engineering. Useful for poetry. However, these do not provide a strict enforcement
mechanism—you can’t be sure that your generated text will
conform to the rules  
  - 5. Using a more powerful model might be an option because such
models are typically better at following instructions  

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
