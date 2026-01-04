# Diffusion LLM-assisted Speech Recognition

This project explores **ASR decoding using a masked diffusion language model** (LLaDA-style), replacing standard left-to-right autoregressive decoding with **iterative denoising of masked tokens** to trade off latency vs accuracy. The approach follows a *compute-limited adaptation* regime: **freeze a strong speech encoder (Whisper) and a strong diffusion LLM mask-predictor**, and **train only a lightweight projector** to align speech features into the diffusion decoder’s embedding space.

---

## Pipeline

### Core idea
- **Frozen Whisper encoder** produces acoustic representations.
- A **trainable MLP projector** maps downsampled speech states to the diffusion decoder embedding dimension.
- A **frozen LLaDA-style masked diffusion decoder** predicts masked transcript tokens in parallel, iteratively refining the sequence via **low-confidence remasking**.

### Inference configurations
This repo supports three decoding modes (same conditioning/prompt; only decoding differs):
1. **Config A — 1-step (single denoise)**: fastest, no self-correction.
2. **Config B — 8-step iterative refinement**: predict + remask low-confidence tokens for 8 steps.
3. **Config C — 8-step + 16 blocks (blockwise diffusion)**: semi-autoregressive left-to-right blocks for stability/controllability on long outputs.

---

## Method overview (high level)

1. **Encode audio**
   - Extract features from waveform and run the **Whisper encoder**.
   - Downsample (e.g., mean-pooling over windows) to reduce the time dimension.

2. **Project to decoder space**
   - Use a small MLP projector: `z = fθ(h̃)` where `z` matches the diffusion decoder embedding size.

3. **Prefix conditioning**
   - Concatenate **speech prefix embeddings** with a **prompt prefix**.
   - The diffusion decoder attends to this prefix while denoising transcript tokens.

4. **Masked diffusion decoding**
   - Initialize the transcript as fully masked `[MASK] × N`.
   - For each step, **fill masked positions** and **re-mask the least confident** tokens to refine.

---

## Results

| Model / Inference | Avg WER (%) | Avg RTFx |
|---|---:|---:|
| Whisper + MLP + AR-LLM (8B, IFT) | 9.65 | 8.34 |
| Whisper + MLP + LLaDA-8B (IFT, 1-step) | 23.90 | 44.15 |
| Whisper + MLP + LLaDA-8B (IFT, 8-step) | 11.14 | 23.40 |
| Whisper + MLP + LLaDA-8B (IFT, 8-step, 16 blocks) | 10.15 | 1.74 |

> **Interpretation:** iterative refinement (8-step) improves WER vs single-step, and blockwise decoding can improve stability and practical speed/controllability trade-offs.

---

## Dataset & metrics
- **Training:** LibriSpeech 100h subset
- **Evaluation:** dev-clean, dev-other, test-clean, test-other
- **Metrics:**
  - **WER** (word error rate)
  - **RTF / RTFx** (real-time factor style speed metric; define normalization clearly in your runs)

---
The following workflow has been tested on an Ubuntu 20.04 LTS machine with 256 GB RAM and an NVIDIA H100 GPU (80GB VRAM).

## Setup & Run

### 0) Install dependencies
1. Create and activate the conda environment:
    ```bash
    conda create -n salm python=3.11.11
    conda activate salm
    pip install -r requirements.txt
    ```
2. Add your HuggingFace token to `./env.sh` and source it:
    ```bash
    source ./env.sh
    ```

### 1) Download datasets (LibriSpeech)
1. Download LibriSpeech splits from: https://www.openslr.org/12  
   - `train-clean-100`
   - `dev-clean`, `dev-other`
   - `test-clean`, `test-other`

2. Create manifest files for all splits (format similar to `./dummy-data/`):
    ```bash
    ./scripts/create_manifest_files.sh
    ```

### 2) Run training
1. Edit the config as needed:
   - `./configs/librispeech/openai-whisper-large-v3_sln-large_llada-8b.yaml`

2. Start training:
    ```bash
    ./scripts/train.sh ./configs/librispeech/openai-whisper-large-v3_sln-large_llada-8b.yaml
    ```

### 3) Run inference
Run inference with the same config:
```bash
./scripts/infer.sh ./configs/librispeech/openai-whisper-large-v3_sln-large_llada-8b.yaml
