from transformers import AutoTokenizer, AutoModelForCausalLM

# Load base model (for prompt baseline)
base_model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load LoRA model
lora_model = AutoModelForCausalLM.from_pretrained("./lora-output")

def generate(model, text):
    prompt = f"Rewrite the following message into a polite customer support response:\n{text}\nResponse:"
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test inputs
test_cases = [
    "We can't help you with this.",
    "Your refund was denied. Nothing we can do.",
    "System is down. Try again later.",
    "That's not our problem.",
    "You entered wrong data."
]

for t in test_cases:
    print("\n==============================")
    print("INPUT:", t)

    print("\nPROMPT OUTPUT:")
    print(generate(base_model, t))

    print("\nLORA OUTPUT:")
    print(generate(lora_model, t))