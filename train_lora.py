import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME  = "distilgpt2"
DATA_PATH   = "dataset_40.jsonl"
OUTPUT_DIR  = "./lora-output"
MAX_LENGTH  = 128
EPOCHS      = 3
BATCH_SIZE  = 4
LR          = 2e-4
FP16        = torch.cuda.is_available()

# ── Load dataset ──────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

raw = load_jsonl(DATA_PATH)

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token     = tokenizer.eos_token
tokenizer.padding_side  = "right"

# ── Label-masked tokenization ─────────────────────────────────────────────────
def tokenize(example):
    prefix = (
        f"Rewrite the following into a polite customer support response:\n"
        f"{example['input']}\nResponse: "
    )
    full_text = prefix + example["output"]

    full   = tokenizer(full_text,  truncation=True, max_length=MAX_LENGTH, padding="max_length")
    prompt = tokenizer(prefix,     truncation=True, max_length=MAX_LENGTH)

    input_ids      = full["input_ids"]
    attention_mask = full["attention_mask"]
    labels         = [-100] * len(prompt["input_ids"]) + input_ids[len(prompt["input_ids"]):]

    # Align to MAX_LENGTH
    labels = labels[:MAX_LENGTH]
    labels += [-100] * (MAX_LENGTH - len(labels))

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }

dataset   = Dataset.from_list(raw)
tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
tokenized.set_format("torch")

# ── Model + LoRA ──────────────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.config.pad_token_id = tokenizer.eos_token_id

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["c_attn"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Training ──────────────────────────────────────────────────────────────────
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    fp16=FP16,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved to {OUTPUT_DIR}")

# ── Inference ─────────────────────────────────────────────────────────────────
def generate(input_text, max_new_tokens=80):
    model.eval()
    prompt = (
        f"Rewrite the following into a polite customer support response:\n"
        f"{input_text}\nResponse:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("Response:")[-1].strip()

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
