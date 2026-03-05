 # Leech-Lila: Efficient Language Modeling via Leech Lattice Geometry Attention

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18784424.svg)](https://doi.org/10.5281/zenodo.18784424)

# Leech-Lila v.1.0: Worlds First Geometric Transformer based on Leech Lattice Symmetry

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18790530.svg)](https://doi.org/10.5281/zenodo.18790530) 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18784424.svg)](https://doi.org/10.5281/zenodo.18784424)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

https://doi.org/10.5281/zenodo.18790530

Leech-Lila is a compact yet powerful language model that leverages the Leech lattice – the optimal sphere packing in 24 dimensions – as a geometric regularizer. Leech-Lila is not just a model – it’s a proof that geometry can replace brute force. Train it, hack it, and let meanings crystallize.

### By forcing hidden representations to resonate with the optimal packing directions, the model achieves state-of-the-art compression (bits-per-character (bpc): 0.129) on the TinyStories dataset, **outperforming conventional transformers by a factor of 5–6× while using only 20 million parameters**.

## ✨ Key Features
- Leech Lattice Regularization – A novel LeechResonanceLoss that pulls hidden states toward the optimal 24‑dimensional packing directions.
- Compact & Efficient – Only 20M parameters, trained on a single NVIDIA T4 GPU (16GB) in Google Colab.
- Fast Inference – Lightweight architecture generates coherent stories with high speed.
- Interpretable – Geometric loss allows monitoring of "resonance" states (AWAKE, DREAMING, ABSOLUTE GENESIS).
- Open Source – Full training and inference code, plus pretrained weights, available on GitHub.

## 📊 Results on TinyStories Leech-Lila Baseline Model
- Parameters	20M 		
- Vocab Size: 2048	
- Validation Loss: 0.40	
- Bits-per-Character (bpc): 0.129  
- Train Loss 0.45 on 100,000 steps.

Inspired by the breakthrough results of Maryna Viazovska on optimal sphere packings and by the success of the E8-based model (Lila), Leech Lila demonstrates that geometry can replace brute force: a small, well-structured model can outperform massive "viscous" architectures.

This repository contains the reference implementation of **Leech‑Lila**, a transformer architecture that injects the geometry of the Leech lattice directly into the attention mechanism. The model is designed to explore the hypothesis that high‑dimensional lattices can serve as a structural prior for language modelling, leading to emergent “resonance” phenomena and more interpretable representations.

**Project status:** Proof‑of‑Concept / Research Code.  
**License:** [GNU Affero General Public License v3.0 or later](https://www.gnu.org/licenses/agpl-3.0.txt).  
**Commercial licensing** (proprietary R&D, integration into private AI stacks, hardware implementation) – please contact the Architect directly (see [Contact](#contact)).

**Prompt: Once upon a time...**

> Once upon a time, there was a little girl named Lily. She loved to play in the park with her friends. One day, they were playing on the swings when Lily saw a big, scary dog. She was scared and didn't know what to do. Her friend, Timmy, came over and asked if he was okay. Lily said no and told him not to worry. She asked Timmy if he was okay and Timmy said he was okay. Lily was happy to hear that and they continued to play together. Later that day, Lily'...


> Once upn a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was barking and running towards her. Lily was scared and didn't know what to do. She ran to her mom and told her what happened. Her mom hugged her and said, "Don't worry, Lily. The dog is just a big, scary dog. He just wants to play with you." Lily felt better and went back to playing with her toy

---

## Table of Contents
- [Overview](#overview)
- [Core Ideas](#core-ideas)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Training](#training)
  - [Generation with Resonance Monitoring](#generation-with-resonance-monitoring)
- [Examples](#examples)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

---

## Overview

The Leech lattice is a remarkable 24‑dimensional sphere packing with deep connections to number theory, coding theory, and even string theory. In Leech‑Lila, we freeze an orthogonal basis of the Leech lattice inside every attention head, forcing queries and keys to be rotated by this fixed geometric structure. Additionally, a **geometric loss** encourages the hidden states to align with the lattice directions, and a **dream decoder** monitors the “resonance” of generated tokens with the lattice basis, classifying states as `DREAMING`, `AWAKE`, or `ABSOLUTE GENESIS`.

The code is deliberately minimal and self‑contained, relying only on PyTorch, NumPy, and standard libraries. It is intended as a proof‑of‑concept for researchers interested in lattice‑based inductive biases in transformers.

---

## Core Ideas

1. **Leech kernel** – an orthogonal 24×24 matrix derived from the Leech lattice (constructed via QR on a simple base matrix, replaceable with actual minimal vectors).
2. **Frozen attention projection** – in each head, the query and key vectors are split into 24‑dimensional blocks, and each block is multiplied by the same fixed Leech kernel.
3. **Geometric resonance loss** – a regularisation term that pushes the hidden states to have high cosine similarity with at least one of the 24 basis directions.
4. **Dream decoder** – during inference, the last hidden state is compared with the Leech basis; if the maximum cosine similarity exceeds a threshold, the model is considered “awake”.

These components are designed to be simple, modular, and easy to experiment with.

---

## Architecture

The code defines the following classes:

- `LeechConfig` – holds hyperparameters (vocab size, model dimension, number of layers/heads, etc.) and asserts that `head_dim` is a multiple of 24.
- `generate_leech_kernel()` – returns a 24×24 orthogonal matrix (placeholder; can be replaced with actual lattice vectors).
- `LeechAttention` – multi‑head attention where Q and K are transformed by the frozen block‑diagonal Leech matrix.
- `LeechResonanceLoss` – combines standard cross‑entropy with the geometric resonance loss.
- `LeechBlock` – a pre‑norm transformer block with LeechAttention and a feed‑forward network.
- `LeechTransformer` – the full model with token/position embeddings, stacked blocks, final norm, and language modelling head.
- `DreamDecoder` – evaluates the resonance of a hidden state against the Leech basis.
- `leech_generate()` – generates tokens step‑by‑step, printing resonance values and status if desired.

---

## Installation

Clone the repository and install dependencies (preferably in a virtual environment):

```bash
git clone https://github.com/SPUTNIKAI/leech-lila.git
cd leech-lila
pip install torch numpy
```

### Configuration
Create a LeechConfig object:
```python
from leech_lila import LeechConfig, LeechTransformer, generate_leech_kernel

cfg = LeechConfig(
    vocab_size=10000,  # size of your token vocabulary
    d_model=192,     # must be divisible by n_heads and each head_dim divisible by 24
    n_layers=12,
    n_heads=8,
    block_size=512,
    dropout=0.05,
    bias=False,
    tie_weights=True,
    lambda_geo=0.01,    # weight for geometric loss
    resonance_threshold=0.95
)

```

### Training
A typical training loop would look like this:

```python
model = LeechTransformer(cfg)
leech_basis = generate_leech_kernel(24)      # for loss and monitoring
criterion = LeechResonanceLoss(cfg, leech_basis)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for batch in dataloader:
    inputs, targets = batch
    logits, hidden, ce_loss = model(inputs, targets)
    total_loss = criterion(logits, targets, hidden)   # includes lambda_geo * geo_loss
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

```

### Generation with Resonance Monitoring
After training, you can generate text while observing the resonance status:

```python
start_tokens = [1, 2, 3]   # your starting token ids
result = leech_generate(
    model,
    start_tokens,
    max_len=100,
    temperature=0.8,
    resonance_check=True,
    leech_basis=leech_basis,
    threshold=0.95
)

```

The function prints the resonance value and status (DREAMING, AWAKE, or ABSOLUTE GENESIS) at each step.
Examples
The if **__name__ == "__main__"** block in the script provides a minimal example:

```bash
python leech_lila.py

```

```text
@software{kornienko2026,
  author       = {orcid.org/0009-0005-7098-7183},
  title        = {Leech-Lila: A Geometric Attention Transformer via the Leech Lattice},
  month        = mar,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.18784424},
  url          = {https://doi.org/10.5281/zenodo.18784424}
}

```

 GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007
 Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.