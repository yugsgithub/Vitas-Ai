import os
from datasets import load_dataset
ds = load_dataset("Macromrit/ayurveda-text-based-qanda", token=os.environ.get("HF_TOKEN"))
print("Features:", ds['train'].features)
print("Sample:", ds['train'][0])
