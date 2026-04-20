import gc
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
OUTPUT_DIR  = "./flan_output_new"
EPOCHS      = 3
BATCH_SIZE  = 4
LR          = 3e-4
FP16        = torch.cuda.is_available()

# ── Load dataset ──────────────────────────────────────────────────────────────
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

raw     = load_jsonl(DATA_PATH)
dataset = Dataset.from_list(raw)

# ── Tokenizer ─────────────────────────────────────────────────────────────────
# T5 has its own pad token (id=0) — do not override it
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── Preprocess ────────────────────────────────────────────────────────────────
# No padding here — DataCollatorForSeq2Seq handles padding + -100 masking
def preprocess(example):
    input_text = (
        example["instruction"]
        + "\n\nMessage:\n"
        + example["input"]
        + "\n\nResponse:"
    )

    model_inputs = tokenizer(
        input_text,
        truncation=True,
        max_length=256,
    )

    labels = tokenizer(
        example["output"],
        truncation=True,
        max_length=128,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess, remove_columns=dataset.column_names)
print(tokenized[0])

# ── Model ─────────────────────────────────────────────────────────────────────
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ── Data collator ─────────────────────────────────────────────────────────────
# label_pad_token_id=-100 ensures padding in labels is ignored by the loss
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
)

# ── Training args ─────────────────────────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
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

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# ── Save (Windows-safe) ───────────────────────────────────────────────────────
gc.collect()
torch.cuda.empty_cache()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved to {OUTPUT_DIR}")

# ── Inference ─────────────────────────────────────────────────────────────────
INSTRUCTION = "Rewrite the message into a polite, professional, and empathetic customer support response."

def generate(input_text, max_new_tokens=80):
    model.eval()
    prompt = f"{INSTRUCTION}\n\nMessage:\n{input_text}\n\nResponse:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

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
