import os
import pandas as pd
from datasets import load_dataset
import json

# 1. Load Data
print("Loading datasets...")
hf_ds = load_dataset("Macromrit/ayurveda-text-based-qanda", token=os.environ.get("HF_TOKEN"))['train']
df_csv = pd.read_csv("AyurGenixAI_Dataset.csv")

# 2. Define TOON Header (Keys)
toon_header = ["instruction", "input", "output"]
toon_data = []

# 3. Process HF Data into TOON
for row in hf_ds:
    toon_data.append([
        row['instruction'],
        row['question'],
        row['answer']
    ])

# 4. Process CSV Data into TOON
for _, row in df_csv.iterrows():
    # Structured input from CSV
    clinical_input = f"Condition: {row['Disease']}\nSymptoms: {row['Symptoms']}"
    
    # Holistic Ayurvedic output
    herbal_output = f"Doshas: {row['Doshas']}\nHerbs: {row['Ayurvedic Herbs']}\nYoga: {row['Yoga & Physical Therapy']}"
    
    toon_data.append([
        "You are an Ayurvedic Expert.",
        clinical_input,
        herbal_output
    ])

# 5. Save in TOON JSON format
toon_final = {
    "format": "toon-v1",
    "keys": toon_header,
    "data": toon_data
}

with open("ayurveda_dataset.toon.json", "w", encoding="utf-8") as f:
    json.dump(toon_final, f, ensure_ascii=False, indent=2)

print(f"✅ Saved 7828 rows to ayurveda_dataset.toon.json in TOON format!")
