# nanochat — AGENTS.md

**Generated:** 2026-03-14  
**Project:** Minimal LLM training harness (GPT-2 in <$100)  
**Type:** Public fork (Andrej Karpathy)  
**Stack:** Python / PyTorch / uv / wandb

---

## OVERVIEW

Simplest experimental harness for training LLMs end-to-end on a single GPU node. Covers tokenization, pretraining, finetuning, evaluation, inference, and chat UI. Train GPT-2 capability (originally $43,000 in 2019) for ~$48 in 2 hours on 8XH100. All hyperparameters auto-calculated from single `--depth` parameter.

**Upstream:** https://github.com/karpathy/nanochat  
**Maintainer:** @karpathy (Andrej Karpathy)

---

## STRUCTURE

```
nanochat/
  README.md              # Main documentation
  pyproject.toml         # uv package config
  uv.lock                # Dependency lock file
  nanochat/              # Core library
    gpt.py               # GPT Transformer nn.Module
    dataloader.py        # Distributed data loader
    optim.py             # AdamW + Muon optimizer
    engine.py            # Inference with KV cache
    tokenizer.py         # BPE tokenizer (GPT-4 style)
    core_eval.py         # DCLM CORE score evaluation
    checkpoint_manager.py # Save/load checkpoints
    ui.html              # Chat frontend (HTML/CSS/JS)
  scripts/
    base_train.py        # Pretraining
    base_eval.py         # Evaluation (CORE, bits per byte)
    chat_sft.py          # SFT fine-tuning
    chat_rl.py           # RL fine-tuning
    chat_web.py          # Web UI server
    chat_cli.py          # CLI chat
    tok_train.py         # Train tokenizer
  runs/
    speedrun.sh          # Train GPT-2 in ~2 hours
    miniseries.sh        # Train model series (d12-d26)
    scaling_laws.sh      # Scaling law experiments
    runcpu.sh            # CPU/MPS example
  tasks/
    arc.py, gsm8k.py, mmlu.py, humaneval.py, smoltalk.py, spellingbee.py
  tests/
    test_engine.py
```

**Total:** 54 Python files, 94 tests

---

## USAGE

### Train GPT-2 (8XH100 node)

```bash
# Full pipeline: train + chat UI (~3 hours)
bash runs/speedrun.sh

# Activate venv and serve chat UI
source .venv/bin/activate
python -m scripts.chat_web
# Visit http://<your-ip>:8000/
```

### Quick Experimentation (d12 model, ~5 min)

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1
```

### Single GPU (gradient accumulation)

```bash
# Omit torchrun, 8x slower but identical results
python -m scripts.base_train --depth=12
```

---

## TIME-TO-GPT-2 LEADERBOARD

| # | Time | val_bpb | CORE | Description | Date | Commit |
|---|------|---------|------|-------------|------|--------|
| 0 | 168h | - | 0.2565 | Original OpenAI GPT-2 | 2019 | - |
| 1 | 3.04h | 0.74833 | 0.2585 | d24 baseline | Jan 29 2026 | 348fbb3 |
| 2 | 2.91h | 0.74504 | 0.2578 | d26 + fp8 | Feb 2 2026 | a67eba3 |
| 3 | 2.76h | 0.74645 | 0.2602 | 1M token batch | Feb 5 2026 | 2c062aa |
| 4 | 2.02h | 0.71854 | 0.2571 | NVIDIA ClimbMix | Mar 4 2026 | 324e69c |
| 5 | **1.80h** | 0.71808 | **0.2690** | autoresearch round 1 | Mar 9 2026 | 6ed7d1d |

**Goal:** Beat GPT-2 CORE score (0.256525) in minimum wall-clock time on 8XH100.

---

## KEY FEATURES

### Single Complexity Dial

```bash
--depth=12  # GPT-1 sized (~5 min training)
--depth=16  # Medium (~15 min)
--depth=24  # GPT-2 sized (~2 hours)
--depth=26  # Slightly larger
```

All other hyperparameters (width, heads, LR, weight decay, training horizon) auto-calculated for compute optimality.

### Precision Management

| Hardware | Default dtype | Override |
|----------|---------------|----------|
| CUDA SM 80+ (A100, H100) | `bfloat16` | `NANOCHAT_DTYPE=float32` |
| CUDA SM < 80 (V100, T4) | `float32` | `NANOCHAT_DTYPE=float16` |
| CPU / MPS | `float32` | - |

```bash
NANOCHAT_DTYPE=bfloat16 torchrun --nproc_per_node=8 -m scripts.base_train
```

### Monitoring (wandb)

1. `val_bpb` vs `step`, `total_training_time`, `total_training_flops`
2. `core_metric` (DCLM CORE score)
3. `train/mfu` (Model FLOPS utilization), `train/tok_per_sec`

---

## NOTES

- **Fork status:** Public reference, active upstream development
- **Hardware:** Optimized for 8XH100 (works on 8XA100, single GPU, CPU/MPS)
- **Cost:** ~$48 for GPT-2 on 8XH100 (~$15 on spot instances)
- **Philosophy:** Minimal, hackable, maximally-forkable "strong baseline"
- **No config monsters:** Single cohesive codebase, no giant config objects
- **AI policy:** Disclose LLM contributions in PRs
- **Related:** Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt)

---

**Last Updated:** 2026-03-14  
**License:** MIT
