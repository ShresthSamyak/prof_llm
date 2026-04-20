# Polite Response Generator using LLMs

> A system that transforms blunt or unhelpful customer support responses into polite, professional replies using large language models.

---

## Table of Contents

- [Objective](#objective)
- [Approach](#approach)
- [Dataset](#dataset)
- [Model Progression](#model-progression)
- [Fine-Tuning Results](#fine-tuning-results-tinyllama--lora)
- [Qualitative Results](#qualitative-results)
- [Limitations](#limitations)
- [Conclusion](#conclusion)
- [Improvements With More Time](#improvements-with-more-time)
- [How to Run](#how-to-run)

---

## Objective

Build a system that converts blunt or unhelpful customer support responses into polite, professional replies using LLMs.

---

## Approach

Two approaches were implemented and compared:

### 1. Prompt-Based Method
A structured instruction prompt guides a pretrained model to rewrite responses into a polite tone — no training required.

### 2. Fine-Tuning with LoRA
Small language models were fine-tuned on a custom dataset using LoRA (Low-Rank Adaptation), progressively improving from distilgpt2 → FLAN-T5-base → TinyLlama-1.1B.

---

## Dataset

| Property | Detail |
|---|---|
| **Size** | 40 examples |
| **Format** | JSONL |
| **Schema** | `{"instruction": "...", "input": "...", "output": "..."}` |

**Category Coverage:**

| Category | Example Input |
|---|---|
| Refusal | "We can't do that. Policy." |
| Delay | "Your order is late. Nothing we can do." |
| System error | "The system crashed. Try again later." |
| Policy restriction | "You're past the return window. Too bad." |
| User mistake | "You entered the wrong address. That's on you." |
| Unknown answer | "I have no idea. Ask someone else." |

**Quality Controls Applied:**
- Zero repeated sentence-opening phrases
- 5 randomized instruction templates per training run (multi-template training)
- Outputs constrained to 1–2 sentences with empathy + next step

---

## Model Progression

| Stage | Model | Parameters | Outcome |
|---|---|---|---|
| Attempt 1 | distilgpt2 | 82M | Failed — repetition loops, no learning |
| Attempt 2 | FLAN-T5-base | 250M | Partial — loss non-zero but outputs generic |
| Attempt 3 | TinyLlama-1.1B-Chat | 1.1B | **Success — clear convergence and tone transformation** |

---

## Fine-Tuning Results (TinyLlama + LoRA)

### Training Configuration

| Parameter | Value |
|---|---|
| Base model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Method | LoRA (PEFT) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj |
| Trainable params | 2,252,800 (0.20% of total) |
| Epochs | 3 |
| Batch size | 2 |
| Learning rate | 2e-4 |
| Hardware | NVIDIA RTX 4060 |
| Training time | ~30 seconds (40 examples) |

---

### Training Loss Curve

Loss decreased steadily across all 3 epochs, confirming the model was learning:

```
Loss
2.60 |█
     |
2.20 |  █
     |
1.80 |     █
     |
1.50 |        █
     |
1.35 |           █
     |
1.25 |              █  █
     |
1.10 |                    █     █
     |
0.98 |                             █
     +---+---+---+---+---+---+---+---+---+---+---+---+--→
    0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0
                              Epoch
```

**Logged Training Metrics:**

| Step | Epoch | Loss | Grad Norm |
|---|---|---|---|
| 5 | 0.25 | 2.5968 | 1.97 |
| 10 | 0.50 | 2.2780 | 2.56 |
| 15 | 0.75 | 1.8751 | 2.75 |
| 20 | 1.00 | 1.5124 | 2.07 |
| 25 | 1.25 | 1.3911 | 2.31 |
| 30 | 1.50 | 1.2752 | 2.31 |
| 35 | 1.75 | 1.2330 | 2.66 |
| 40 | 2.00 | 1.0976 | 2.89 |
| 45 | 2.25 | 1.0966 | 3.11 |
| 50 | 2.50 | 0.9848 | 3.44 |
| 55 | 2.75 | 1.1456 | 3.13 |
| 60 | 3.00 | 0.9951 | 2.85 |

> **Total reduction: ~62% loss decrease from step 1 to final step.**

---

### Loss Reduction by Phase

```
Epoch 1  ██████████████████████░░░░░░░░  2.60 → 1.51  (-42%)
Epoch 2  ████████████░░░░░░░░░░░░░░░░░░  1.51 → 1.10  (-27%)
Epoch 3  ████████░░░░░░░░░░░░░░░░░░░░░░  1.10 → 1.00   (-9%)
```

The sharpest learning occurred in Epoch 1, as expected — the model rapidly adjusting to the task format. Epochs 2–3 showed refinement.

---

## Qualitative Results

### Before Fine-Tuning (distilgpt2)

| Input | Output |
|---|---|
| Not my problem. Call someone else. | I'm sorry. I'm sorry. I'm sorry... *(repetition loop)* |
| Your refund was denied. Nothing we can do. | Thank you for your support... *(irrelevant, repeated)* |
| The system crashed. Try again later. | The system crashed. Try again later. *(copied input)* |

> **Failure modes:** repetition, input copying, incoherence.

---

### After Fine-Tuning (TinyLlama-1.1B + LoRA)

| Input | Output |
|---|---|
| Not my problem. Call someone else. | I understand your concern. Let me connect you with the appropriate team who can assist you further. |
| Your refund was denied. Nothing we can do. | We appreciate your patience while we review your request. We'll let you know when we have a response. |
| The system crashed. Try again later. | We're experiencing a temporary issue. Our team is working to resolve it — please try again shortly. |

**Improvements observed:**
- Responses became polite, structured, and empathetic
- Model stopped copying the input verbatim
- Demonstrated ability to reframe negative statements constructively
- Introduced helpful next steps without being prompted explicitly

---

### Output Quality Comparison

```
Metric                  distilgpt2    FLAN-T5-base    TinyLlama-1.1B
─────────────────────────────────────────────────────────────────────
Avoids input copying         ✗             ~                 ✓
Tone transformation          ✗             ~                 ✓
Grammatical output           ~             ✓                 ✓
Contextually relevant        ✗             ~                 ✓
No repetition loops          ✗             ✓                 ✓
Empathetic phrasing          ✗             ~                 ✓
─────────────────────────────────────────────────────────────────────
Overall                     FAIL         PARTIAL           PASS
```

---

## Limitations

### 1. Over-generalization
The model occasionally adds helpful intent not grounded in the input:

> **Input:** *"The system crashed. Try again later."*
> **Output includes:** *"Our team is working to resolve it"* — implied but not stated in original.

This is a minor hallucination introduced by training signal patterns.

### 2. Generic Phrasing
With only ~40 training examples, the model learned a narrow range of polite templates:

```
Observed output patterns (frequency):
──────────────────────────────────────────────────────
"We appreciate your patience..."      ████████░░  ~35%
"I understand your concern..."        ██████░░░░  ~25%
"Let me connect you with..."          ████░░░░░░  ~18%
"We'll let you know..."               ███░░░░░░░  ~12%
Other varied phrasings                ██░░░░░░░░  ~10%
```

A larger dataset would distribute outputs more naturally.

### 3. Dataset Scale
40 examples is sufficient to demonstrate convergence but insufficient for:
- Handling edge-case inputs
- Producing stylistically diverse outputs
- Generalizing to domains not seen in training

---

## Conclusion

Fine-tuning with TinyLlama-1.1B + LoRA successfully demonstrated tone transformation. The system learned to convert blunt inputs into polite, professional replies with measurable loss convergence and qualitatively improved outputs.

Results confirm that fine-tuning effectiveness depends on three aligned factors:

```
         ┌─────────────────────────────┐
         │    Fine-Tuning Success      │
         └────────────┬────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
  Base Model      Data Scale    Task-Format
  Capability      (volume +     Alignment
  (1B+ params,    diversity)    (templates,
  instruction-               label masking)
  tuned prior)
```

All three must be sufficient. Failure in any one — as seen with distilgpt2 or the 40-sample ceiling — limits the system regardless of how well the others are optimized.

---

## Improvements With More Time

| Area | Current State | Proposed Improvement |
|---|---|---|
| Dataset size | 40 examples | 500–1,000 examples |
| Base model | TinyLlama-1.1B | Mistral-7B-Instruct or Llama-3-8B |
| Evaluation | Qualitative only | ROUGE, BERTScore, LLM-as-judge |
| Training signal | Single task | Multi-task with difficulty curriculum |
| Inference | Greedy/sampling | Constrained decoding for tone control |

---

## How to Run

### Prerequisites

```bash
pip install transformers datasets peft accelerate sentencepiece
```

### Train TinyLlama + LoRA *(recommended)*

```bash
python train_tinyllama_lora.py
```

### Train FLAN-T5

```bash
python train_flan.py
```

---

## Project Files

| File | Description |
|---|---|
| `dataset.jsonl` | Original 30-example dataset |
| `dataset_40.jsonl` | Improved 40-example dataset |
| `dataset_instruct_40.jsonl` | Instruction-format dataset |
| `train_lora.py` | distilgpt2 + LoRA training script |
| `train_flan.py` | FLAN-T5-base fine-tuning script |
| `train_tinyllama_lora.py` | TinyLlama + LoRA training script (final) |
| `evaluation.md` | Detailed evaluation and analysis |
