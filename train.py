from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# 1. Configuration
# Using the pre-quantized 4-bit version to save 5GB of download and start faster
model_id = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
max_seq_length = 2048
load_in_4bit = True

# 2. Load Tokenizer and Model
# Using 'use_fast=True' and 'low_cpu_mem_usage=True' for speed
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)


# 3. Prepare for Training and Add LoRA Adapters
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# 4. Data Preparation
prompt_style = """<|user|>
{}<|end|>
<|assistant|>
{}<|end|>"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["question"]
    outputs      = examples["answer"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        full_input = f"{instruction}\n{input}" if instruction else input
        text = prompt_style.format(full_input, output)
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("json", data_files="merged_ayurveda_dataset.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 5. Training Setup
trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        max_length = max_seq_length,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "paged_adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        remove_unused_columns = False,
    ),
)

# 6. Train
trainer.train()

# 7. Save the model
model.save_pretrained("ayurveda_lora_model")
tokenizer.save_pretrained("ayurveda_lora_model")
print("Training complete! Model saved to 'ayurveda_lora_model'")

