import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "google/flan-t5-base"
DATA_PATH   = "dataset_instruct_40.jsonl"
OUTPUT_DIR  = "./flan-output"
MAX_INPUT   = 128
MAX_TARGET  = 64
EPOCHS      = 3
BATCH_SIZE  = 4
LR          = 3e-4
FP16        = torch.cuda.is_available()

# ── Load dataset ──────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def format_input(ex):
    return f"{ex['instruction']}\n\nMessage:\n{ex['input']}\n\nResponse:"

raw     = load_jsonl(DATA_PATH)
dataset = Dataset.from_list([{"source": format_input(ex), "target": ex["output"]} for ex in raw])

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    model_inputs = tokenizer(
        batch["source"],
        max_length=MAX_INPUT,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        batch["target"],
        max_length=MAX_TARGET,
        truncation=True,
        padding="max_length",
    )
    # Replace padding token id with -100 so it's ignored in loss
    label_ids = [
        [(t if t != tokenizer.pad_token_id else -100) for t in seq]
        for seq in labels["input_ids"]
    ]
    model_inputs["labels"] = label_ids
    return model_inputs

tokenized = dataset.map(tokenize, batched=True, remove_columns=["source", "target"])
tokenized.set_format("torch")

# ── Model ─────────────────────────────────────────────────────────────────────
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ── Training ──────────────────────────────────────────────────────────────────
args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    fp16=FP16,
    predict_with_generate=True,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved to {OUTPUT_DIR}")

# ── Inference ─────────────────────────────────────────────────────────────────
def generate(instruction, input_text, max_new_tokens=80):
    model.eval()
    prompt = f"{instruction}\n\nMessage:\n{input_text}\n\nResponse:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ── Test samples ──────────────────────────────────────────────────────────────
INSTRUCTION = "Rewrite the message into a polite, professional, and empathetic customer support response."

samples = [
    "Not my problem. Call someone else.",
    "Your refund was denied. Nothing we can do.",
    "The system crashed. Try again later.",
]

print("\n── Inference test ──")
for s in samples:
    print(f"\nInput:  {s}")
    print(f"Output: {generate(INSTRUCTION, s)}")
