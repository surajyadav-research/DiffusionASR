# Param 1

**BharatGen** introduces **Param 1**, a bilingual language model pretrained from scratch on English and Hindi. With 2.9 billion parameters, it serves as a powerful foundational model for text completion task.

**Param1-2.9B** outperforms leading models like **LLaMA-3.2B**, **Gemma-2B**, **Granite-2B**, and **Granite-3B** on various standard benchmarks. 

This early release is only the **Pre-Trained** checkpoint equipped with inference support via **NVIDIA NeMo**.

---

## 📁 Folder Structure

```
Param1/
├── model_extracted/       # Extracted model checkpoints/configuration files
├── model.nemo             # Original .nemo packaged model file
├── nemo_inference.sh      # Shell script for running inference 
└── README.md              # This documentation file
```

---

## 🐳 Docker Setup

### 1. Pull the Docker Image
```bash
docker pull kundeshwar/kp_lm_eval_oom_issue
```

### 2. Create a Docker Container
Replace `/path_to_your_project` and `path_to_your_workspace` accordingly:
```bash
docker run --name name_of_your_container --gpus all -it -d -v /path_to_your_project:path_to_your_workspace kundeshwar/kp_lm_eval_oom_issue
```

---

## 🚀 Model Inference

### Steps to run inference using the `.nemo` file:

1. Locate the `model.nemo` file and copy the path.
2. Open `nemo_inference.sh` and replace the placeholder path with the path to your `.nemo` file:
```bash
gpt_model_file="/path_to_.nemo_file"
```

---

## 📊 Benchmarks (zero-shot)

| Task | Param 1 | Gemma2-2B (PT) | llama3.2-3B (distill PT) | granite-3.1-2B (PT) | granite-3.1-3B (PT) | qwen-2.5-3B (PT) |
|------|------------------------|-------------------|-----------------------------|--------------------|--------------------|--------------------|
| ARC Challenge | 46.7 | 49.7 | 46.0 | 47.2 | 45.2 | 47.4 |
| ARC Easy | 74.6 | 80.3 | 71.7 | 76.8 | 75.8 | 73.2 |
| HellaSwag | 71.4 | 73.0 | 73.7 | 75.5 | 72.6 | 73.6 |
| HellaSwag Hi | 44.1 | 38.6 | 40.0 | 31.0 | 28.5 | 32.9 |
| MMLU En | 41.4 | 47.1 | 53.9 | 47.8 | 41.0 | 64.9 |
| MMLU Hi | 30.7 | 30.0 | 35.0 | 29.0 | 25.7 | 38.32 |
| PIQA | 79.3 | 78.3 | 77.31 | 79.4 | 78.2 | 78.84 |
| TriviaQA | 38.5 | 32.9 | 50.83 | 26.2 | 27.5 | 42.27 |
| TruthfulQA - Gen (BLEU) | 38.2 | 29.7 | 21.8 | 34.0 | 36.7 | 36.96 |
| TruthfulQA - MC1 Acc | 28.0 | 24.0 | 25.3 | 26.1 | 26.4 | 32.07 |
| TruthfulQA - MC2 Acc | 43.8 | 36.2 | 39.2 | 39.0 | 39.9 | 48.95 |
| SuperGLUE - boolq | 70.6 | 73.7 | 72.7 | 71.0 | 68.5 | 77.27 |
| SuperGLUE - rte | 62.5 | 61.7 | 54.5 | 69.3 | 54.9 | 75.09 |
| SuperGLUE - WiC | 49.5 | 49.5 | 50.0 | 50.3 | 52.3 | 61.75 |
| SuperGLUE - multirc | 56.9 | 55.9 | 57.2 | 57.2 | 57.2 | 39.52 |

> **Notes:**
> - **PT**: Pre-Trained  
> - **en-hi**: English-Hindi  
> - Pre-trained on **7.5 Trillion tokens**  

---

## 🧠 Model Architecture

- Hidden size: 2048  
- Intermediate size: 7168  
- Number of attention heads: 16  
- Number of hidden layers: 32  
- Number of key-value heads: 8  
- Maximum position embeddings: 2048  
- Activation function: **SiLU**  
- Positional embeddings: **Rotary (RoPE)** with `rope_theta=10000.0`  
- Attention: **Grouped-query attention**  
- Precision: **bf16-mixed**

---

## 🏗️ Training Details

- **Training Infrastructure**: Yotta’s Shakti Cloud  
- **Hardware**: NVIDIA H100 – 512 GPUs  
- **Framework**: NVIDIA NeMo

---

Important Guidelines for Early Checkpoint Release of Our LLM

1. :construction: Early Development Status
* This model is in the initial phase of Pre-training.
* It is yet to undergo supervised fine-tuning, safety alignment, or rigorous evaluation.
* The release is intended to showcase progress, gather feedback, and encourage research and experimentation.
* Outputs may at times be incoherent, irrelevant, or of suboptimal quality.

2. :books: Data Sources and Potential Artifacts
* To preserve the Model's understanding on the global front, part of the training Data also includes data crawled from the Internet hence it may contain inherited artifacts;
* Due to the increased prevalence of AI-generated content online in the current times, the model may occasionally mimic such statements and incorrectly identify itself.
* These artifacts are natural consequences of using publicly available data found on the internet although critical but important since such data is important for the model to build a global know-how and we will be addressing issues like this in future iterations of the current Model.

3. :lock: Lack of Alignment and Guardrails
* A preliminary-level alignment or safety mechanisms have been implemented at this stage.
* The model is yet to under go full-scale instruction tuning, supervised fine-tuning, or reinforcement learning from human feedback (RLHF).
* As a result, it may occasionally: 
	* Generate biased, offensive, or unsafe content
	* Be susceptible to misuse or prompt injection (jailbreaking)
	* Respond to harmful or unethical prompts without refusal
* This model must not be deployed in any production without reading Intent use section.

4. :microscope: Intended Use
* This release is provided exclusively for research, experimentation and contribution to the open source community.
* Suggested use cases include: 
	* Assessing early-stage LLM behavior
	* Debugging model training pipelines and configurations
	* Benchmarking or custom fine-tuning by the community
* Access to early-checkpoint should embibe a sense of motivation and enthusiasm among the open source community to take such early-stage check point and build India-Specific Innovative use cases on top of it. This should also help foster innovation among the Community.

5. :x: Not Intended For
* This model is not suitable for real-world or commercial deployment.
* It should not be used for:
  * Customer service, content generation, or summarization
  * Applications involving factual accuracy, safety, or user interaction
  * Producing public-facing content without human oversight

6. :scroll: Licensing and Responsibility
* Released under an open license with responsible usage guidelines.
* License: MIT
* Users are expected to:
  * Adhere to ethical usage practices and legal regulations
  * Avoid malicious or unsafe deployment
  * Credit the authors as per the licensing terms

7. <img src="https://flagcdn.com/w40/in.png" width="20"/> Acknowledgement of Origin  
* A home-grown effort initiated in India with limited resources.
* This work represents a bottom-up initiative to develop LLMs from scratch within India.
* It reflects our humble, resource-constrained journey to contribute meaningfully to the open-source AI ecosystem.
* We hope to foster collaboration and growth within the broader community.

8. :test_tube: Transparency & Community Collaboration
* We welcome contributions and open dialogue.
* We encourage the community to share feedback, report issues, and collaborate.
* Future versions will introduce better alignment, improved training scale, and more curated datasets.
* Together, we aim to evolve toward safer and more capable AI systems.

---


## 📜 License

This model is released under the **BharatGen non-commercial license**.  
Please refer to the [LICENSE](./LICENSE) for terms and conditions.

---
