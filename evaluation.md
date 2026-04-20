# Evaluation: Prompt-Based vs LoRA Fine-Tuned Response Rewriting

## Qualitative Comparison

The prompt-based method produced coherent, empathetic outputs across all three examples. Responses were grammatically correct, contextually appropriate, and required no training.

The LoRA fine-tuned model (distilgpt2) failed to generalize: outputs were either repetitive loops, echoed the input verbatim, or produced incoherent text unrelated to the task.

## Key Observations

- Prompt engineering works out of the box on capable base models and scales well to varied inputs.
- distilgpt2 is too small (82M parameters) to learn a stylistic transformation from 40 examples — it lacks the representational capacity to generalize.
- LoRA did not cause the failure; the base model did. LoRA is effective when the base model already understands language well enough to be steered.

## Trade-offs

| | Prompt Engineering | LoRA Fine-Tuning |
|---|---|---|
| Setup time | Minutes | Hours |
| Compute | Inference-only | GPU training required |
| Consistency | Depends on prompt | Can be enforced via training |
| Works on small models | No | No |
| Data required | None | Yes |

## What Failed and Why

The fine-tuned model collapsed into repetition or input copying — classic signs of an underpowered model overfitting a tiny dataset without learning the underlying transformation. 40 examples are insufficient for a model that has no prior instruction-following ability.

### Additional Experiment

A second training attempt was conducted with improved formatting:

- Instruction-style prompts
- Label masking (training only on target output)

Despite these improvements, the model still failed to produce reliable transformations.

### Improved Dataset Experiment

A third iteration introduced a structured instruction-tuning format:

- Explicit `instruction` + `input` + `output` schema
- Increased linguistic diversity with no repeated templates
- Avoided overuse of generic phrases (e.g., "I'm sorry")
- Covered multiple support scenarios: refusal, delay, errors, policy, user mistakes

Despite these improvements, the fine-tuned model continued to struggle, confirming that performance limitations stem from base model capability rather than dataset quality alone.

### Insight

This experiment demonstrates that:
- Improving dataset quality alone is not sufficient for reliable fine-tuning
- Small, non-instruction-tuned models (e.g., distilgpt2) lack the prior needed for stylistic transformation tasks
- Prompt-based methods can outperform fine-tuning in low-data, low-compute settings

Successful fine-tuning depends on both sufficient training data and a base model with meaningful instruction-following capacity.

## Improvements With More Time

- Replace distilgpt2 with `google/flan-t5-base` or `TinyLlama-1.1B` — both have instruction-following priors that LoRA can effectively steer.
- Expand dataset to 500+ examples with greater input diversity.
- Add a dedicated evaluation split and score with ROUGE or BERTScore to quantify improvement over training.
