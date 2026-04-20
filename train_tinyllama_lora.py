import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH   = "dataset_instruct_40.jsonl"
OUTPUT_DIR  = "./tinyllama-lora-output"
MAX_LENGTH  = 512
EPOCHS      = 3
BATCH_SIZE  = 2
LR          = 2e-4

# ── Load dataset ──────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

raw = load_jsonl(DATA_PATH)

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token     = tokenizer.eos_token
tokenizer.padding_side  = "right"

# ── Format + label masking ────────────────────────────────────────────────────
def preprocess(example):
    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n"
    )
    full_text = prompt + example["output"] + tokenizer.eos_token

    full_ids   = tokenizer(full_text,   truncation=True, max_length=MAX_LENGTH)["input_ids"]
    prompt_ids = tokenizer(prompt,      truncation=True, max_length=MAX_LENGTH)["input_ids"]

    prompt_len = len(prompt_ids)
    labels     = [-100] * prompt_len + full_ids[prompt_len:]

    # Pad to MAX_LENGTH
    pad_len    = MAX_LENGTH - len(full_ids)
    input_ids  = full_ids  + [tokenizer.pad_token_id] * pad_len
    labels     = labels    + [-100]                   * pad_len
    attn_mask  = [1]       * len(full_ids) + [0]      * pad_len

    return {
        "input_ids":      input_ids[:MAX_LENGTH],
        "attention_mask": attn_mask[:MAX_LENGTH],
        "labels":         labels[:MAX_LENGTH],
    }

dataset   = Dataset.from_list(raw)
tokenized = dataset.map(preprocess, remove_columns=dataset.column_names)
tokenized.set_format("torch")

# ── Model + LoRA ──────────────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto",
)
model.config.use_cache = False

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)
model.enable_input_require_grads()
model.print_trainable_parameters()

# ── Training ──────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    fp16=True,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
    gradient_checkpointing=True,
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved to {OUTPUT_DIR}")

# ── Inference ─────────────────────────────────────────────────────────────────
INSTRUCTION = "Rewrite the message into a polite, professional, and empathetic customer support response."

def generate(input_text, max_new_tokens=120):
    model.eval()
    prompt = (
        f"### Instruction:\n{INSTRUCTION}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Response:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens, not the prompt
    generated = output[0][input_len:]
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    # Stop at any next section marker
    return decoded.split("###")[0].strip()

# ── Test samples ──────────────────────────────────────────────────────────────
samples = [
    "Not my problem. Call someone else.",
    "Your refund was denied. Nothing we can do.",
    "The system crashed. Try again later.",
]

print("\n── Inference test ──")
for s in samples:
    print(f"\nInput:  {s}")
    print(f"Output: {generate(s)}")
