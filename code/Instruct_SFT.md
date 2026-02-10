## Supervised fine-tuning (SFT) for instruction models — how it actually works

SFT instruction tuning teaches a model **how to respond given a prompt shape**, and choosing a small, stable Instruct base like Qwen2.5 lets you control that behavior cheaply, predictably, and visibly using LoRA.

### What SFT *is* 

Supervised fine-tuning trains a language model to **continue a prompt with a preferred response** using standard next-token prediction.

For each training example, the model sees a **single concatenated sequence**:

```
[prompt tokens] + [response tokens]
```

Training objective:

* At each position *t*, predict token *t+1*
* Loss is computed over the entire sequence (or sometimes only the response span)

There is **no new objective** and **no reinforcement signal**.
SFT is ordinary cross-entropy training, just with *instruction-shaped text*.

---

## SFT instruction tuning vs prompting — what is fundamentally different

Prompting steers a frozen model per request; SFT rewires the model so the desired instruction-following behavior becomes the default across all prompts.


### Core distinction

* **Prompting** changes the *input* at inference time
* **SFT instruction tuning** changes the *model parameters* at training time

Everything else follows from this.

---

## Prompting (system / instruction / context)

### What happens

At inference, you supply additional tokens:

```text id="x6t4q"
<System>
<Instruction>
<Context>
<User prompt>
```

The model computes:

```text id="k7p9s"
P(next_token | all_prompt_tokens, frozen_weights)
```

The weights **do not change**.

---

### What prompting can do well

* Control tone and verbosity
* Inject temporary context
* Specify output format
* Steer style *for this call only*
* Apply guardrails and constraints

Prompting is:

* Cheap
* Reversible
* Stateless
* Per-request

---

### Hard limits of prompting

Prompting **cannot**:

* Permanently change behavior
* Reduce generic filler reliably
* Fix instruction drift across tasks
* Enforce consistency across prompts
* Remove unwanted modes (e.g. over-verbosity)

Every new call must restate constraints.

---

## SFT instruction tuning

### What happens

During training, the model repeatedly sees:

```text id="4rj2v"
Instruction:
...
Response:
<ideal output>
```

Weights are updated so that:

```text id="9n1cp"
P(desired_response | instruction_pattern) ↑
```

This alters:

* Token preferences
* Response length distributions
* Formatting defaults
* Task priors

The model internalizes the behavior.

---

### What SFT does that prompting cannot

SFT can:

* Make concise answers the *default*
* Remove boilerplate globally
* Normalize output structure
* Improve instruction adherence under weak prompts
* Reduce need for system prompts

After SFT, you can prompt minimally:

```text id="a0x4m"
Summarize this document.
```

…and still get the tuned behavior.

---

## Distributional view (why this matters)

Prompting:

```text id="b9w2q"
Shift input → same conditional distribution family
```

SFT:

```text id="h4m8s"
Shift conditional distribution itself
```

Prompting steers *within* the model’s learned space.
SFT reshapes that space.

---

## Memory and persistence

| Aspect                      | Prompting   | SFT       |
| --------------------------- | ----------- | --------- |
| Persistence                 | None        | Permanent |
| Scope                       | Single call | All calls |
| Requires restating rules    | Yes         | No        |
| Affects defaults            | No          | Yes       |
| Survives context truncation | No          | Yes       |

---

## Why prompting feels “almost enough” (but isn’t)

Modern instruct models are already heavily SFT-trained.
Prompting *works* because you are exploiting **existing priors**.

But when you want:

* Consistent structure
* Minimal verbosity
* Domain-specific phrasing
* Reliable instruction adherence

Prompting hits a ceiling.

---

## When prompting is sufficient

Use prompting when:

* Behavior varies per request
* You want fast iteration
* Constraints are lightweight
* You don’t control deployment

---

## When SFT is justified

Use SFT when:

* You want a new default behavior
* Prompts repeat across calls
* You see the same errors again and again
* You want shorter, cleaner prompts
* You are shipping a system, not chatting

---

## Instruction SFT

### Why instruction formatting matters

The model does not “understand” instructions as a special concept.
It learns **statistical regularities** such as:

* “When text looks like an instruction header → a helpful response follows”
* “After `Response:` → produce structured, task-specific output”

If your formatting is inconsistent, the model learns noise.

That is why:

* Stable delimiters (`Instruction:`, `Input:`, `Response:`)
* Consistent ordering
* Minimal variation

matter more than dataset size at small scales.

---

### What changes during SFT

SFT shifts:

* **Conditional probability mass**
* **Token preferences**
* **Response style and structure**

It does **not**:

* Add new knowledge reliably
* Improve reasoning depth
* Fix hallucinations
* Teach long-horizon planning

SFT teaches *how to respond*, not *how to think*.

---

## Instruct models vs base models

### Base model

* Trained on raw text
* Learns language statistics
* No preference for “helpful answers”
* Will continue *anything* plausibly

### Instruct model

* Base model + SFT (and usually RLHF)
* Learned conversational structure
* Learned refusal patterns
* Learned verbosity and politeness norms

When you fine-tune:

* **Base → Instruct**: you must teach *everything* (format, behavior, safety)
* **Instruct → Domain-specific instruct**: you refine behavior, not invent it

For small-budget LoRA work, **always start from an Instruct model**.

---

## What SFT is doing in the Qwen2.5 walkthrough

In the walkthrough:

```text
Instruction:
<task description>

Response:
<ideal answer>
```

The model is trained so that:

```
P(token | "Instruction: ... Response:") 
```

is shifted toward:

* More precise answers
* More consistent formatting
* Less generic filler
* More domain-specific vocabulary

LoRA ensures:

* Base weights stay frozen
* Only a small number of low-rank matrices are updated
* Training is cheap and reversible

---

## Why Qwen2.5 is a good SFT base

### Architectural reasons

* Modern decoder-only transformer
* Clean projection layers (`q_proj`, `v_proj`) → LoRA-friendly
* Stable training under 4-bit quantization

### Practical reasons

* Small variants (0.5B / 1.5B) fit Colab
* Good tokenizer coverage
* Strong instruction baseline
* Minimal boilerplate

### Teaching / debugging reasons

* Loss curves behave predictably
* Overfitting is visible quickly
* Changes in output are obvious after a few hundred steps

---

## How to choose a base model for SFT (general rule)

### Step 1 — Decide *what you are changing*

* Style / structure → Instruct model
* Domain vocabulary → Instruct model
* New reasoning skills → SFT is the wrong tool

### Step 2 — Match model size to data size

* <10k examples → ≤2B parameters
* 10k–100k → 3B–7B
* More data does **not** compensate for bad formatting

### Step 3 — Check LoRA compatibility

Model must have:

* Named projection layers
* Stable HF implementation
* Known working LoRA configs

### Step 4 — Quantization tolerance

If it breaks in 4-bit → skip it for Colab work.

---

## What SFT will *not* fix (important)

Do **not** expect SFT to:

* Reduce hallucinations reliably
* Improve factual accuracy on unseen topics
* Learn tools or planning
* Handle distribution shift

Those require:

* Better data
* Better prompting
* Retrieval
* Evaluation under shift
* Or different training objectives

------------

## LoRA fine-tuning template (Colab) — Qwen2.5-0.5B / 1.5B Instruct

### 0) Install dependencies

```bash
!pip -q install -U "transformers>=4.39.0" "datasets>=2.18.0" "accelerate>=0.27.0" \
  "peft>=0.10.0" "trl>=0.8.6" "bitsandbytes>=0.43.0" "sentencepiece" "evaluate"
```

---

### 1) Pick model + basic config

**Explanation:**

* `0.5B` runs on smaller GPUs and is faster.
* `1.5B` needs more VRAM; use 4-bit quantization (QLoRA) to fit in Colab GPUs.

```python
import os, torch
from dataclasses import dataclass

# Choose one:
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

OUTPUT_DIR = "./qwen25_lora_out"
SEED = 42

# Training set size knobs (keep small for demos)
MAX_TRAIN_SAMPLES = 5000
MAX_EVAL_SAMPLES  = 500
```

---

### 2) Load model in 4-bit (QLoRA) + tokenizer

**Explanation:**

* QLoRA = load base model in 4-bit and train small LoRA adapters on top.
* This is the default approach for Colab.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
)

print("Loaded:", MODEL_NAME)
```

---

### 3) Dataset: quick SFT dataset (instruction → response)

**Explanation:**
You need examples in the form:

* `instruction` (or `prompt`)
* `response` (or `output`)

Below uses a built-in public dataset and normalizes it into a single `text` field.

```python
from datasets import load_dataset

# A small, common instruction dataset; swap if you want
raw = load_dataset("tatsu-lab/alpaca", split="train")

# Optional: downsample for speed
raw = raw.shuffle(seed=SEED)
raw_train = raw.select(range(min(MAX_TRAIN_SAMPLES, len(raw))))
raw_eval  = raw.select(range(min(MAX_EVAL_SAMPLES, len(raw))))

def format_alpaca(ex):
    # Alpaca has: instruction, input, output
    instruction = ex["instruction"].strip()
    inp = (ex["input"] or "").strip()
    output = ex["output"].strip()

    if inp:
        prompt = f"Instruction:\n{instruction}\n\nInput:\n{inp}\n\nResponse:\n"
    else:
        prompt = f"Instruction:\n{instruction}\n\nResponse:\n"

    # For SFT, you want model to learn to produce output after the prompt
    text = prompt + output
    return {"text": text}

train_ds = raw_train.map(format_alpaca, remove_columns=raw_train.column_names)
eval_ds  = raw_eval.map(format_alpaca, remove_columns=raw_eval.column_names)

train_ds[0]
```

---

### 4) Tokenization

**Explanation:**
We tokenize the combined `text`. `labels` are set to the same tokens so the model is trained to predict the next token.

```python
MAX_LEN = 512  # increase if you have VRAM headroom

def tokenize_batch(batch):
    out = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )
    out["labels"] = out["input_ids"].copy()
    return out

train_tok = train_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
eval_tok  = eval_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
```

---

### 5) Attach LoRA adapters

**Explanation:**
LoRA trains low-rank matrices on top of key projection layers.
For Qwen-family causal LMs, `q_proj` and `v_proj` is a good default target.

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],  # good default for many decoder LMs
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

---

### 6) Train (Transformers Trainer)

**Explanation:**

* `per_device_train_batch_size` + `gradient_accumulation_steps` controls effective batch size.
* `fp16/bf16` depends on GPU support.
* `evaluation_strategy="steps"` gives quick feedback.

```python
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    seed=SEED,
    num_train_epochs=1,  # increase for real training (e.g., 2-3)
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    logging_steps=20,
    save_steps=200,
    eval_steps=200,
    evaluation_strategy="steps",
    save_strategy="steps",
    report_to="none",
    bf16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,  # A100 etc.
    fp16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8,   # T4/V100
    optim="paged_adamw_8bit",
    max_grad_norm=1.0,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    data_collator=data_collator,
)

trainer.train()
```

---

### 7) Save LoRA adapters (+ tokenizer)

**Explanation:**
This saves only the adapters (small), not a full merged model.

```python
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Saved adapters to:", OUTPUT_DIR)
```

---

### 8) Inference with the fine-tuned adapters

**Explanation:**
Reload base model + attach adapters, then generate.

```python
from peft import PeftModel

# Reload base model (still 4-bit) and attach LoRA adapters
base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
)

ft = PeftModel.from_pretrained(base, OUTPUT_DIR)
ft.eval()

def generate(prompt, max_new_tokens=200, temperature=0.2):
    inputs = tokenizer(prompt, return_tensors="pt").to(ft.device)
    with torch.no_grad():
        out = ft.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

test_prompt = "Instruction:\nWrite a concise Python function to compute rolling z-score.\n\nResponse:\n"
print(generate(test_prompt))
```

---

### 9) Optional: merge LoRA into the base model (makes a standalone model)

**Explanation:**
Merging creates a single set of weights. This usually requires loading the base model in higher precision (not 4-bit).
For Colab, you may skip merging and just keep adapters.

```python
# Optional: merging generally needs full precision base model.
# If you want to try (may OOM on small GPUs):

# base_fp16 = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     device_map="auto",
#     torch_dtype=torch.float16,
# )
# ft_fp16 = PeftModel.from_pretrained(base_fp16, OUTPUT_DIR)
# merged = ft_fp16.merge_and_unload()
# merged.save_pretrained("./qwen25_merged")
# tokenizer.save_pretrained("./qwen25_merged")
```

---

## Common adjustments (copy/paste)

### If you hit OOM

```python
# Reduce sequence length and/or batch, increase accumulation
MAX_LEN = 384
# args.per_device_train_batch_size = 1
# args.gradient_accumulation_steps = 16
```

### If training is unstable

```python
# Lower learning rate, increase warmup
# args.learning_rate = 1e-4
# args.warmup_ratio = 0.06
```

### If LoRA target modules don’t match

**Explanation:**
Some architectures name projection layers differently. Print module names and adjust.

```python
# Inspect module names to choose targets
for name, _ in model.named_modules():
    if any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"]):
        print(name)
        break
```
