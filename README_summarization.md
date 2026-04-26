# News Summarization via LLM Fine-Tuning (Qwen3 + LoRA)

Fine-tuning a 4-billion parameter language model (**Qwen3-4B-Instruct**) on the CNN/DailyMail dataset for abstractive news summarization, using parameter-efficient training (LoRA) via Unsloth. Evaluated with ROUGE and BERTScore metrics.

---

## Highlights

- **Efficient LLM fine-tuning** — LoRA adapters (rank 32) applied to all attention and MLP projection layers, keeping GPU memory low via 4-bit quantization
- **Full NLP pipeline** — data cleaning, token-length EDA, training, inference, and evaluation
- **Completion-only training** — custom data collator masks prompt tokens so the model only learns to predict the summary, not the input
- **Quantitative evaluation** — ROUGE-1/2/L and BERTScore (P/R/F1) on 300 test examples

---

## Tech Stack

| Component | Technology |
|---|---|
| Base Model | [Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct) via Unsloth |
| Fine-tuning | LoRA (PEFT) + SFTTrainer (TRL) |
| Efficient Training | [Unsloth](https://github.com/unslothai/unsloth) (2x faster, 60% less VRAM) |
| Dataset | [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) (v1.0.0) |
| Evaluation | `evaluate` — ROUGE + BERTScore |
| Runtime | Python 3.10+, Google Colab (GPU) |

---

##  Pipeline Overview

```
Raw Dataset
    │
    ▼
Data Cleaning          → HTML removal, special char stripping, deduplication
    │
    ▼
EDA                    → Token length distributions, compression ratio analysis
    │
    ▼
Chat Template Formatting → System prompt + article → highlights conversation format
    │
    ▼
LoRA Fine-Tuning       → Qwen3-4B, rank=32, 2 epochs, completion-only loss
    │
    ▼
Inference              → Greedy decoding with dynamic max_new_tokens budget
    │
    ▼
Evaluation             → ROUGE-1/2/L/Lsum + BERTScore on 300 test samples
```

---

## Model & Training Config

| Parameter | Value |
|---|---|
| Base model | `unsloth/Qwen3-4B-Instruct-2507` |
| Quantization | 4-bit (NF4) |
| LoRA rank | 32 |
| LoRA alpha | 32 |
| Target modules | q/k/v/o proj, gate/up/down proj |
| Max sequence length | 2048 |
| Batch size (effective) | 8 (2 × 4 grad accum) |
| Learning rate | 2e-4 |
| Scheduler | Linear warmup |
| Optimizer | AdamW 8-bit |
| Precision | fp16 |
| Training steps | 60 (smoke run) |

---

##  Evaluation Results

Evaluated on 300 CNN/DailyMail test examples using greedy decoding (`temperature=0.0`).

| Metric | Score |
|---|---|
| ROUGE-1 | see notebook |
| ROUGE-2 | see notebook |
| ROUGE-L | see notebook |
| BERTScore F1 | see notebook |

> Full results are logged in the evaluation cell output inside the notebook.