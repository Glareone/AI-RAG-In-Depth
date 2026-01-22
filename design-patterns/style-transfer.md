# Pattern 3: Style Transfer

<img width="784" height="901" alt="image" src="https://github.com/user-attachments/assets/e8b712cd-f6a5-4804-b268-6a2d532406ee" />

## Quick Summary

**Problem**: Convert content from one style to another when you have paired examples (before → after).

**Solution**: Use in-context learning (few-shot) or fine-tuning on paired examples (A → B).

**Key Advantage**: Style Transfer works with **in-context learning and few-shot strategy** - no fine-tuning required if you have good examples.

**WHEN to choose WHAT**:
   * **Few-shot only (3-5 examples)** → Use in-context learning (fastest, no training)
   * **More examples available (100+)** → Use fine-tuning (better quality)
   * **API models** → Use few-shot or API fine-tuning
   * **Open-source models** → Use few-shot or LoRA fine-tuning

**WHAT to choose WHEN**:
  * **Choose Style Transfer when**: You have A -> B paired examples
  * **Choose Reverse Neutralization when**: You only have B, need to generate new A's

**Remember**: Style Transfer = Direct A→B conversion (with or without training)

---

## When to Use: Style Transfer vs Reverse Neutralization

### From Examples Perspective

| What You Have | Pattern to Use |
|---------------|----------------|
| **Paired examples** (A → B)<br>Example: "Casual note → Professional email" | **Style Transfer**<br>Few-shot ICL or fine-tuning |
| **Only styled examples** (B only)<br>Example: "Your personal emails (no source versions)" | **Reverse Neutralization**<br>Create neutral versions, fine-tune |

### From Mechanism Perspective

| Approach | Style Transfer | Reverse Neutralization |
|----------|---------------|----------------------|
| **In-Context Learning** | ✅ **Possible** (3-5 shot examples) | ❌ Not possible |
| **Fine-Tuning** | ✅ Optional (for better quality) | ✅ **Required** (SFT/DPO/RLHF) |
| **Use Case** | Convert existing content A→B | Generate NEW content in style B |
| **Training Data** | ✅ Ready (paired examples) | ❌ Must create (neutralize first) |

### Decision Guide

```
Do you have paired examples (before → after styling)?
├─ YES → Use Style Transfer
│   ├─ Few examples (3-5) → In-context learning (few-shot)
│   └─ Many examples (100+) → Fine-tuning
└─ NO → Use Reverse Neutralization (fine-tuning required)
```

### Key Differences

| Aspect | Style Transfer | Reverse Neutralization |
|--------|---------------|----------------------|
| Training data | ✅ Ready (paired A→B) | ❌ Must create (neutralize + reverse) |
| Inference steps | 1 (direct A→B) | 2 (generate + convert) |
| ICL/Few-shot | ✅ **Works great** | ❌ Not possible |
| Fine-tuning | Optional (quality boost) | Required (only option) |
| Cost | Lower (1 call) | Higher (2 calls) |

**Example Scenarios**:
- ✅ **Style Transfer**: "Convert my casual notes to professional emails" (have both versions)
- ✅ **Reverse Neutralization**: "Generate Lufthansa complaint in MY style" (new topic, only have old emails)

---

## How It Works

### Approach 1: In-Context Learning (Few-Shot)

**No training required** - provide 3-5 examples in the prompt:

```
Convert this casual note to professional email.

Examples:
Casual: "hey can we meet tomorrow?"
Professional: "Dear colleague, I would like to schedule a meeting tomorrow."

Casual: "thanks for the help!"
Professional: "Thank you for your valuable assistance."

Now convert: "got ur message, lets talk later"
```

**When to use**: Quick solutions, API models, limited data (3-10 examples)

### Approach 2: Fine-Tuning

Train model on paired examples for consistent quality:

```
Training Pair: {input: casual_text, output: professional_text}
                       ↓
                 Fine-tuned Model
                       ↓
              Converts any casual → professional
```

**When to use**: High volume, consistent quality needed, 100+ examples available

---

## Real Example: Casual Notes → Professional Emails

**What you have**: 10 pairs of casual notes and their professional email versions

**What you need**: Convert any new casual note to professional email

### Few-Shot Approach (In-Context Learning)

```python
from openai import OpenAI

client = OpenAI()

# Few-shot prompt with examples
prompt = """Convert casual notes to professional emails.

Examples:
Casual: "hey john, can we push the meeting?"
Professional: "Dear John, Would it be possible to reschedule our meeting?"

Casual: "thx for sending that doc!"
Professional: "Thank you for providing the document."

Casual: "got ur email, will review asap"
Professional: "I have received your email and will review it promptly."

Now convert this:
Casual: "meeting went well, lets followup next week"
Professional:"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
# Output: "The meeting was productive. I suggest we schedule a follow-up for next week."
```

**Advantages**:
- ✅ No training required
- ✅ Works with any API model
- ✅ Can update style by changing examples
- ✅ Fast to implement (minutes)

### Fine-Tuning Approach (For Better Quality)

```python
# 1. Prepare training data (100+ pairs)
training_data = [
    {"input": "hey john, can we push the meeting?",
     "output": "Dear John, Would it be possible to reschedule our meeting?"},
    {"input": "thx for sending that doc!",
     "output": "Thank you for providing the document."},
    # ... 100+ more pairs
]

# 2. Fine-tune (Azure OpenAI example)
# Upload JSONL → Create fine-tuning job → Deploy model

# 3. Use fine-tuned model
response = client.chat.completions.create(
    model="ft:gpt-3.5-turbo:your-model-id",
    messages=[{"role": "user", "content": "meeting went well, lets followup next week"}]
)
# Output: "The meeting was productive. I suggest we schedule a follow-up for next week."
```

**Advantages**:
- ✅ More consistent quality
- ✅ Handles edge cases better
- ✅ No prompt engineering needed

---

## Implementation: Few-Shot vs Fine-Tuning

### Few-Shot (In-Context Learning)

**Recommended for**:
- Quick prototypes
- Limited examples (3-10 pairs)
- API models (GPT-4, Claude)
- Changing requirements

**Implementation**:
```python
# Works with any API model
prompt = f"""Convert style A to B.

Examples:
{example_1}
{example_2}
{example_3}

Convert: {input_text}"""
```

**Cost**: Per-token pricing only (no training cost)

---

### Fine-Tuning

#### API Models (OpenAI, Azure, Anthropic)

**Recommended: SFT**

```python
# Azure OpenAI format
{
  "messages": [
    {"role": "system", "content": "Convert casual to professional"},
    {"role": "user", "content": "hey can we meet?"},
    {"role": "assistant", "content": "Would it be possible to schedule a meeting?"}
  ]
}
```

**Cost**: ~$8-12 per 1M tokens for training

---

#### Open-Source Models (Llama, Mistral)

**Recommended: LoRA/QLoRA**

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")

# Configure LoRA (only trains 1-2% of weights)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

# Fine-tune on paired examples (casual → professional)
model = get_peft_model(model, lora_config)
```

**Cost**: ~$5-20 (GPU rental for few hours)

---

### Decision Matrix

| Your Setup | Recommended Approach | When to Use | Cost |
|------------|---------------------|-------------|------|
| **API models (GPT-4/Claude)** | **Few-shot ICL** | 3-10 examples | Per-token only |
| **API models + many examples** | SFT via API | 100+ examples, need consistency | $8-12 per 1M tokens |
| **Llama (local GPU)** | **Few-shot ICL** | Quick testing | Free |
| **Llama + many examples** | LoRA fine-tuning | 100+ examples, best quality | $5-20 |

**Key Takeaway**:
- 🔵 **Start with few-shot ICL** → Fast, no training, works great
- 🟢 **Use fine-tuning** → When you need consistent high quality with 100+ examples

---

## Image Style Transfer

Style Transfer works for images too - convert photos to artistic styles.

### Approach 1: Few-Shot with Vision Models

```python
# Use GPT-4V or Claude with example images
prompt = """Convert this photo to Van Gogh style.

Example:
[Original Photo] → [Van Gogh Style]

Now convert: [User's Photo]"""
```

### Approach 2: Fine-Tuning (LoRA for Stable Diffusion)

```python
# Fine-tune Stable Diffusion on artist's paintings
# Use LoRA adapters for style (efficient, ~100MB per style)

from diffusers import StableDiffusionPipeline
import torch

# Load base model + LoRA adapter for Van Gogh style
pipeline = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5")
pipeline.load_lora_weights("van-gogh-style-lora")

# Generate image in Van Gogh style
image = pipeline("portrait of a person", guidance_scale=7.5).images[0]
```

**When to use**:
- Few-shot: Quick experiments with vision models
- Fine-tuning: Consistent style, multiple images, production use

---

## Considerations

1. **Few-shot works surprisingly well** - Always try ICL first before fine-tuning
2. **Examples quality matters** - Use diverse, representative pairs
3. **Context window limits** - API models: ~10 examples, fine-tuning: unlimited
4. **Fine-tuning trade-offs** - Better quality but slower iteration, training cost
5. **Style consistency** - Few-shot may vary, fine-tuning is more consistent

---

## Summary: When to Choose What

```
Style Transfer Decision Tree:

Do you have paired examples (A → B)?
├─ YES → Style Transfer
│   ├─ Have 3-10 examples → Few-shot ICL (recommended start)
│   ├─ Have 100+ examples → Consider fine-tuning (for consistency)
│   ├─ Using API model → Few-shot ICL (easiest)
│   └─ Need consistent quality → Fine-tune (SFT or LoRA)
└─ NO → Use Reverse Neutralization (see Pattern 4)
```

**Remember**: Style Transfer's superpower is **in-context learning** - you can get great results with just 3-5 examples and no training.