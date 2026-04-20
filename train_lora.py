import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "distilgpt2"
DATA_PATH    = "dataset_40.jsonl"
OUTPUT_DIR   = "./lora-output"
MAX_LENGTH   = 128
EPOCHS       = 3
BATCH_SIZE   = 4
LR           = 2e-4
FP16         = torch.cuda.is_available()

# ── Load dataset ─────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def format_example(ex):
    return f"Rewrite this into a polite customer support response:\n{ex['input']}\nResponse: {ex['output']}"

raw = load_jsonl(DATA_PATH)
dataset = Dataset.from_list([{"text": format_example(ex)} for ex in raw])

# ── Tokenizer ────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized.set_format("torch")

# ── Model + LoRA ─────────────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

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
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved to {OUTPUT_DIR}")

# ── Inference ─────────────────────────────────────────────────────────────────
def generate(input_text, max_new_tokens=80):
    model.eval()
    prompt = f"Rewrite this into a polite customer support response:\n{input_text}\nResponse:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
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
