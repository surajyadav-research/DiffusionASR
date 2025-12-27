---
language:
- bn
- en
- gu
- hi
- kn
- ml
- mr
- or
- pa
- ta
- te
library_name: transformers
---


# Sarvam-1

Sarvam-1 is a 2-billion parameter language model specifically optimized for Indian languages. It provides best in-class performance in 10 Indic languages (bn, gu, hi, kn, ml, mr, or, pa, ta, te) when compared with popular models like Gemma-2-2B and Llama-3.2-3B. It is also competitive against the much larger models like Llama-3.1-8B in these languages. More details can be found in our [release blog](https://www.sarvam.ai/blogs/sarvam-1).

The model was trained with  [NVIDIA NeMo™ Framework](https://github.com/NVIDIA/NeMo)  on the Yotta Shakti Cloud using HGX H100 systems.

*Note: This is a text-completion model. It is meant to be finetuned on downstream tasks, and cannot be used directly as a chat or an instruction-following model.*

## Key Features

- **Optimized for 10 Indian Languages**: Built from the ground up to support major Indian languages alongside English
- **Superior Token Efficiency**: Achieves fertility rates of 1.4-2.1 across all supported languages, 2-4x more efficient than existing multilingual models
- **High-Quality Training Data**: Trained on a curated corpus of ~4 trillion tokens with 2 trillion high-quality Indic tokens
- **Efficient Inference**: 4-6x faster inference compared to larger models while matching or exceeding their performance on Indic language tasks

## Model Architecture

- Hidden size: 2048
- Intermediate size: 11,008
- Number of attention heads: 16
- Number of hidden layers: 28
- Number of key-value heads: 8
- Maximum position embeddings: 8,192
- Activation function: SwiGLU
- Positional embeddings: Rotary (RoPE) with theta=10,000
- Training: Grouped-query attention and bfloat16 mixed-precision

## Performance

### Translated Academic Benchmarks (Zero-shot)

- MMLU: 44.44
- ARC-Challenge: 58.50
- TriviaQA: 90.62
- BoolQ: 80.68

### IndicGenBench (One-shot)

- Flores English-to-Indic translation: 39.83 chrF++
- CrossSum: 20.48 chrF++
- XORQA: 25.27 F1
- XQUAD: 41.58 F1

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("sarvamai/sarvam-1")
tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-1")

# Example usage
text = "कर्नाटक की राजधानी है:"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=5)
result = tokenizer.decode(outputs[0])
```

## Training Details

- Training Infrastructure: Yotta's Shakti cluster
- Hardware: 1,024 GPUs
- Training Duration: 5 days
- Framework: NVIDIA NeMo

## License

Sarvam non-commercial license: See the [LICENSE](LICENSE.md) file

## Acknowledgements

- NVIDIA: for support with the NeMo codebase
- Yotta: for sccess to the Shakti GPU cluster
- AI4Bharat: for their academic partnership and expertise in Indian language technologies