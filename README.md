 # Leech LILA 24D PoC

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18784424.svg)](https://doi.org/10.5281/zenodo.18784424)

# Leech-Lila: Geometric Transformer via Leech Lattice

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18784424.svg)](https://doi.org/10.5281/zenodo.18784424)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

This repository contains the reference implementation of **Leech‑Lila**, a transformer architecture that injects the geometry of the Leech lattice directly into the attention mechanism. The model is designed to explore the hypothesis that high‑dimensional lattices can serve as a structural prior for language modelling, leading to emergent “resonance” phenomena and more interpretable representations.

**Project status:** Proof‑of‑Concept / Research Code.  
**License:** [GNU Affero General Public License v3.0 or later](https://www.gnu.org/licenses/agpl-3.0.txt).  
**Commercial licensing** (proprietary R&D, integration into private AI stacks, hardware implementation) – please contact the Architect directly (see [Contact](#contact)).

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