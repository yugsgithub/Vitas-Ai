"""
Ayurveda LoRA Model Evaluation Script
Generates training curves, inference metrics, ROUGE scores, and perplexity.
"""

import json, math, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge_score import rouge_scorer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_MODEL   = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
LORA_MODEL   = "./ayurveda_lora_model"
TRAINER_LOG  = "./outputs/checkpoint-60/trainer_state.json"
DATASET_FILE = "merged_ayurveda_dataset.jsonl"
OUT_DIR      = "./eval_results"
NUM_SAMPLES  = 30   # samples used for ROUGE / perplexity
MAX_NEW_TOKENS = 150

os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = {
    "bg":      "#0f1117",
    "panel":   "#1a1d2e",
    "accent1": "#7c6bff",
    "accent2": "#00d4aa",
    "accent3": "#ff6b6b",
    "accent4": "#ffa654",
    "text":    "#e2e8f0",
    "grid":    "#2a2d3e",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["panel"],
    "axes.edgecolor":    PALETTE["grid"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["text"],
    "ytick.color":       PALETTE["text"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["grid"],
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
})

# ─────────────────────────────────────────────
# 1. LOAD TRAINING LOG
# ─────────────────────────────────────────────
print("📊 Loading training logs...")
with open(TRAINER_LOG) as f:
    state = json.load(f)
logs = [e for e in state["log_history"] if "loss" in e]

steps     = [e["step"]                for e in logs]
loss      = [e["loss"]                for e in logs]
lr        = [e["learning_rate"]       for e in logs]
grad_norm = [e["grad_norm"]           for e in logs]
entropy   = [e["entropy"]             for e in logs]
token_acc = [e["mean_token_accuracy"] for e in logs]
num_tok   = [e["num_tokens"]          for e in logs]

# Smoothed loss (EMA)
def ema(vals, alpha=0.3):
    s = [vals[0]]
    for v in vals[1:]:
        s.append(alpha * v + (1 - alpha) * s[-1])
    return s

loss_smooth = ema(loss)

# ─────────────────────────────────────────────
# 2. PLOT TRAINING DASHBOARD
# ─────────────────────────────────────────────
print("🎨 Plotting training dashboard...")
fig = plt.figure(figsize=(20, 14), facecolor=PALETTE["bg"])
fig.suptitle("Ayurveda LoRA — Training Dashboard", fontsize=18,
             fontweight="bold", color=PALETTE["text"], y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

def styled_ax(ax, title):
    ax.set_title(title, fontsize=11, fontweight="bold",
                 color=PALETTE["text"], pad=8)
    ax.grid(True, alpha=0.4)
    ax.spines[:].set_color(PALETTE["grid"])

# ── 2a. Training Loss
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(steps, loss, color=PALETTE["accent1"], alpha=0.3, lw=1.2, label="Raw")
ax1.plot(steps, loss_smooth, color=PALETTE["accent1"], lw=2.2, label="EMA")
ax1.fill_between(steps, loss_smooth, alpha=0.12, color=PALETTE["accent1"])
ax1.set_xlabel("Step"); ax1.set_ylabel("Loss")
ax1.legend(framealpha=0.2, facecolor=PALETTE["panel"])
styled_ax(ax1, "Training Loss")

# ── 2b. Learning Rate Schedule
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(steps, lr, color=PALETTE["accent4"], lw=2)
ax2.fill_between(steps, lr, alpha=0.15, color=PALETTE["accent4"])
ax2.set_xlabel("Step"); ax2.set_ylabel("LR")
styled_ax(ax2, "Learning Rate Schedule")

# ── 2c. Gradient Norm
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(steps, grad_norm, color=PALETTE["accent3"], lw=1.8)
ax3.set_xlabel("Step"); ax3.set_ylabel("Grad Norm")
styled_ax(ax3, "Gradient Norm")

# ── 2d. Token Accuracy
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(steps, token_acc, color=PALETTE["accent2"], lw=1.8)
ax4.fill_between(steps, token_acc, alpha=0.12, color=PALETTE["accent2"])
ax4.set_ylim(0, 1); ax4.set_xlabel("Step"); ax4.set_ylabel("Accuracy")
styled_ax(ax4, "Mean Token Accuracy")

# ── 2e. Entropy
ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(steps, entropy, color="#e879f9", lw=1.8)
ax5.set_xlabel("Step"); ax5.set_ylabel("Entropy")
styled_ax(ax5, "Prediction Entropy")

# ── 2f. Cumulative Tokens
ax6 = fig.add_subplot(gs[2, 0])
cum_tok = np.cumsum(num_tok)
ax6.plot(steps, cum_tok / 1e3, color=PALETTE["accent4"], lw=1.8)
ax6.fill_between(steps, cum_tok / 1e3, alpha=0.12, color=PALETTE["accent4"])
ax6.set_xlabel("Step"); ax6.set_ylabel("Tokens (K)")
styled_ax(ax6, "Cumulative Tokens Seen")

# ── 2g. Loss vs Token Acc correlation
ax7 = fig.add_subplot(gs[2, 1])
sc = ax7.scatter(loss, token_acc, c=steps, cmap="plasma", s=40, alpha=0.85)
plt.colorbar(sc, ax=ax7, label="Step")
ax7.set_xlabel("Loss"); ax7.set_ylabel("Token Accuracy")
styled_ax(ax7, "Loss vs Token Accuracy")

# ── 2h. Step Summary Box
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis("off")
summary = [
    ("Final Loss",        f"{loss[-1]:.4f}"),
    ("Min Loss",          f"{min(loss):.4f}"),
    ("Final Token Acc",   f"{token_acc[-1]:.4f}"),
    ("Max Token Acc",     f"{max(token_acc):.4f}"),
    ("Final Entropy",     f"{entropy[-1]:.4f}"),
    ("Final Grad Norm",   f"{grad_norm[-1]:.5f}"),
    ("Total Steps",       str(steps[-1])),
    ("Total Tokens (K)",  f"{cum_tok[-1]/1e3:.1f}"),
    ("Train Loss (avg)",  f"{state.get('log_history',-1)[-1].get('train_loss', 'N/A')}"),
]
for i, (k, v) in enumerate(summary):
    y = 0.92 - i * 0.1
    ax8.text(0.05, y, k, transform=ax8.transAxes,
             fontsize=9.5, color="#a0aec0")
    ax8.text(0.65, y, v, transform=ax8.transAxes,
             fontsize=9.5, fontweight="bold", color=PALETTE["accent2"])
ax8.set_title("Training Summary", fontsize=11, fontweight="bold",
              color=PALETTE["text"], pad=8)

plt.savefig(f"{OUT_DIR}/training_dashboard.png", dpi=150,
            bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close()
print(f"  ✅ Saved: {OUT_DIR}/training_dashboard.png")

# ─────────────────────────────────────────────
# 3. LOAD MODEL FOR INFERENCE
# ─────────────────────────────────────────────
print("\n🤖 Loading model for inference...")
tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(base, LORA_MODEL)
model.eval()
print("  ✅ Model loaded.")

PROMPT_TMPL = "<|user|>\n{}<|end|>\n<|assistant|>\n"

def generate(question, instruction="You are a world class Ayurvedic expert. Answer the query using your Ayurvedic knowledge."):
    prompt = PROMPT_TMPL.format(f"{instruction}\n{question}")
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
    return decoded.strip()

def compute_perplexity(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc.input_ids.to(model.device)
    with torch.no_grad():
        loss = model(input_ids=input_ids, labels=input_ids).loss
    return math.exp(loss.item())

# ─────────────────────────────────────────────
# 4. ROUGE + PERPLEXITY ON TEST SAMPLES
# ─────────────────────────────────────────────
print(f"\n📐 Evaluating on {NUM_SAMPLES} samples (ROUGE + Perplexity)...")
dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
samples  = dataset.select(range(NUM_SAMPLES))

scorer_rouge = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)

r1_list, r2_list, rL_list, ppl_list = [], [], [], []
generated_texts = []

for i, row in enumerate(samples):
    ref  = row["answer"]
    pred = generate(row["question"])
    generated_texts.append({"question": row["question"], "reference": ref, "prediction": pred})

    scores = scorer_rouge.score(ref, pred)
    r1_list.append(scores["rouge1"].fmeasure)
    r2_list.append(scores["rouge2"].fmeasure)
    rL_list.append(scores["rougeL"].fmeasure)

    ppl = compute_perplexity(pred if pred else ref)
    ppl_list.append(min(ppl, 200))  # cap outliers for plotting

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{NUM_SAMPLES} done...")

# ─────────────────────────────────────────────
# 5. PLOT INFERENCE METRICS DASHBOARD
# ─────────────────────────────────────────────
print("\n🎨 Plotting inference metrics...")
fig2, axes = plt.subplots(2, 3, figsize=(20, 11), facecolor=PALETTE["bg"])
fig2.suptitle("Ayurveda LoRA — Inference Metrics", fontsize=18,
              fontweight="bold", color=PALETTE["text"], y=0.98)

sample_ids = list(range(1, NUM_SAMPLES + 1))

# ── ROUGE-1
ax = axes[0][0]
ax.bar(sample_ids, r1_list, color=PALETTE["accent1"], alpha=0.85)
ax.axhline(np.mean(r1_list), color="white", lw=1.5, ls="--",
           label=f"Mean: {np.mean(r1_list):.3f}")
ax.set_xlabel("Sample"); ax.set_ylabel("Score"); ax.set_ylim(0, 1)
ax.legend(framealpha=0.2, facecolor=PALETTE["panel"])
styled_ax(ax, "ROUGE-1")

# ── ROUGE-2
ax = axes[0][1]
ax.bar(sample_ids, r2_list, color=PALETTE["accent2"], alpha=0.85)
ax.axhline(np.mean(r2_list), color="white", lw=1.5, ls="--",
           label=f"Mean: {np.mean(r2_list):.3f}")
ax.set_xlabel("Sample"); ax.set_ylabel("Score"); ax.set_ylim(0, 1)
ax.legend(framealpha=0.2, facecolor=PALETTE["panel"])
styled_ax(ax, "ROUGE-2")

# ── ROUGE-L
ax = axes[0][2]
ax.bar(sample_ids, rL_list, color=PALETTE["accent4"], alpha=0.85)
ax.axhline(np.mean(rL_list), color="white", lw=1.5, ls="--",
           label=f"Mean: {np.mean(rL_list):.3f}")
ax.set_xlabel("Sample"); ax.set_ylabel("Score"); ax.set_ylim(0, 1)
ax.legend(framealpha=0.2, facecolor=PALETTE["panel"])
styled_ax(ax, "ROUGE-L")

# ── Perplexity
ax = axes[1][0]
ax.plot(sample_ids, ppl_list, color=PALETTE["accent3"], lw=2, marker="o", ms=4)
ax.axhline(np.mean(ppl_list), color="white", lw=1.5, ls="--",
           label=f"Mean: {np.mean(ppl_list):.2f}")
ax.set_xlabel("Sample"); ax.set_ylabel("Perplexity")
ax.legend(framealpha=0.2, facecolor=PALETTE["panel"])
styled_ax(ax, "Perplexity (generated text)")

# ── ROUGE Distribution (violin)
ax = axes[1][1]
parts = ax.violinplot([r1_list, r2_list, rL_list], positions=[1,2,3],
                       showmedians=True, showextrema=True)
colors = [PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent4"]]
for pc, c in zip(parts["bodies"], colors):
    pc.set_facecolor(c); pc.set_alpha(0.7)
parts["cmedians"].set_color("white")
parts["cmins"].set_color(PALETTE["grid"])
parts["cmaxes"].set_color(PALETTE["grid"])
parts["cbars"].set_color(PALETTE["grid"])
ax.set_xticks([1,2,3]); ax.set_xticklabels(["ROUGE-1","ROUGE-2","ROUGE-L"])
ax.set_ylabel("Score"); ax.set_ylim(0, 1)
styled_ax(ax, "ROUGE Score Distribution")

# ── Metric Summary Table
ax = axes[1][2]
ax.axis("off")
metrics = {
    "ROUGE-1 Mean":    f"{np.mean(r1_list):.4f}",
    "ROUGE-1 Max":     f"{np.max(r1_list):.4f}",
    "ROUGE-2 Mean":    f"{np.mean(r2_list):.4f}",
    "ROUGE-L Mean":    f"{np.mean(rL_list):.4f}",
    "Perplexity Mean": f"{np.mean(ppl_list):.2f}",
    "Perplexity Min":  f"{np.min(ppl_list):.2f}",
    "Samples Eval":    str(NUM_SAMPLES),
    "Final Train Loss":f"{loss[-1]:.4f}",
    "Max Token Acc":   f"{max(token_acc):.4f}",
}
for i, (k, v) in enumerate(metrics.items()):
    y = 0.92 - i * 0.1
    ax.text(0.05, y, k, transform=ax.transAxes, fontsize=9.5, color="#a0aec0")
    ax.text(0.65, y, v, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", color=PALETTE["accent2"])
ax.set_title("Inference Summary", fontsize=11, fontweight="bold",
             color=PALETTE["text"], pad=8)

plt.savefig(f"{OUT_DIR}/inference_metrics.png", dpi=150,
            bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close()
print(f"  ✅ Saved: {OUT_DIR}/inference_metrics.png")

# ─────────────────────────────────────────────
# 6. SAMPLE Q&A OUTPUT
# ─────────────────────────────────────────────
print("\n🔬 Sample Generations:")
for i in range(min(5, NUM_SAMPLES)):
    g = generated_texts[i]
    print(f"\n[{i+1}] Q: {g['question'][:80]}...")
    print(f"     REF : {g['reference'][:120]}...")
    print(f"     PRED: {g['prediction'][:120]}...")

# ─────────────────────────────────────────────
# 7. SAVE FULL METRICS JSON
# ─────────────────────────────────────────────
results = {
    "training": {
        "final_loss":      loss[-1],
        "min_loss":        min(loss),
        "final_token_acc": token_acc[-1],
        "max_token_acc":   max(token_acc),
        "final_entropy":   entropy[-1],
        "total_steps":     steps[-1],
        "train_loss_avg":  state["log_history"][-1].get("train_loss"),
    },
    "inference": {
        "num_samples":     NUM_SAMPLES,
        "rouge1_mean":     float(np.mean(r1_list)),
        "rouge1_max":      float(np.max(r1_list)),
        "rouge2_mean":     float(np.mean(r2_list)),
        "rougeL_mean":     float(np.mean(rL_list)),
        "perplexity_mean": float(np.mean(ppl_list)),
        "perplexity_min":  float(np.min(ppl_list)),
    },
    "samples": generated_texts,
}
with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ All done! Results saved to '{OUT_DIR}/'")
print(f"   • {OUT_DIR}/training_dashboard.png")
print(f"   • {OUT_DIR}/inference_metrics.png")
print(f"   • {OUT_DIR}/metrics.json")
print("\n📊 Final Metrics:")
print(f"   ROUGE-1 : {np.mean(r1_list):.4f}")
print(f"   ROUGE-2 : {np.mean(r2_list):.4f}")
print(f"   ROUGE-L : {np.mean(rL_list):.4f}")
print(f"   PPL Mean: {np.mean(ppl_list):.2f}")
