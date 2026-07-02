import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Shubh769/ayurveda-phi3-lora is a LoRA adapter, not a full model.
# We load the base model first and then apply the LoRA adapter.
model_id = "microsoft/Phi-3-mini-4k-instruct"
adapter_id = "Shubh769/ayurveda-phi3-lora"

print(f"Loading base model: {model_id}...")

# Phi-3 is natively supported in transformers 5.x. 
# DO NOT use trust_remote_code=True for the model loading as the remote modeling_phi3.py 
# in the Microsoft repo has compatibility issues with newer transformers (KeyError: 'type').
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)

# Load base model in bfloat16 (Memory efficient and works well on Apple Silicon GPU)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",      # Uses MPS (Mac GPU) automatically
    trust_remote_code=False # Use native transformers implementation
)

from peft import PeftModel
print(f"Applying LoRA adapter: {adapter_id}...")
model = PeftModel.from_pretrained(model, adapter_id)

model.eval()
print("Model and Adapter loaded successfully!")

# Quick inference test
prompt = "What are the benefits of Ashwagandha in Ayurveda?"
# Standard Phi-3 Chat Template
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

print("\nGenerating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=256, 
        do_sample=True, 
        temperature=0.7,
        top_p=0.9
    )

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
print("\nPrompt:", prompt)
print("\nResponse:", response)