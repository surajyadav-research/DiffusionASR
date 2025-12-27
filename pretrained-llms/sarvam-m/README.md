---
library_name: transformers
license: apache-2.0
language:
- en
- bn
- hi
- kn
- gu
- mr
- ml
- or
- pa
- ta
- te
base_model:
- mistralai/Mistral-Small-3.1-24B-Base-2503
base_model_relation: finetune
---

# Sarvam-M
<p align="center">
  <a href="https://dashboard.sarvam.ai/playground"
     target="_blank" rel="noopener noreferrer">
    <img
      src="https://img.shields.io/badge/🚀 Chat on Sarvam&nbsp;Playground-1488CC?style=for-the-badge&logo=rocket"
      alt="Chat on Sarvam Playground"
    />
  </a>
</p>


# Model Information

`sarvam-m` is a multilingual, hybrid-reasoning, text-only language model built on Mistral-Small. This post-trained version delivers exceptional improvements over the base model:

- +20% average improvement on Indian language benchmarks
- +21.6% enhancement on math benchmarks
- +17.6% boost on programming benchmarks

Performance gains are even more impressive at the intersection of Indian languages and mathematics, with an outstanding +86% improvement in romanized Indian language GSM-8K benchmarks.

Learn more about sarvam-m in our detailed [blog post](https://www.sarvam.ai/blogs/sarvam-m).

# Key Features

- **Hybrid Thinking Mode**: A single versatile model supporting both "think" and "non-think" modes. Use the think mode for complex logical reasoning, mathematical problems, and coding tasks, or switch to non-think mode for efficient, general-purpose conversation.

- **Advanced Indic Skills**: Specifically post-trained on Indian languages alongside English, embodying a character that authentically reflects and emphasizes Indian cultural values.

- **Superior Reasoning Capabilities**: Outperforms most similarly-sized models on coding and math benchmarks, demonstrating exceptional reasoning abilities.

- **Seamless Chatting Experience**: Full support for both Indic scripts and romanized versions of Indian languages, providing a smooth and accessible multilingual conversation experience.

# Quickstart 

The following code snippet demonstrates how to use `sarvam-m` using Transformers. 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "sarvamai/sarvam-m"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)

# prepare the model input
prompt = "Who are you and what is your purpose on this planet?"

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(**model_inputs, max_new_tokens=8192)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
output_text = tokenizer.decode(output_ids)

if "</think>" in output_text:
    reasoning_content = output_text.split("</think>")[0].rstrip("\n")
    content = output_text.split("</think>")[-1].lstrip("\n").rstrip("</s>")
else:
    reasoning_content = ""
    content = output_text.rstrip("</s>")

print("reasoning content:", reasoning_content)
print("content:", content)
```

> [!NOTE]
> For thinking mode, we recommend `temperature=0.5`; for no-think mode, `temperature=0.2`.


# With Sarvam APIs

```python
from openai import OpenAI

base_url = "https://api.sarvam.ai/v1"
model_name = "sarvam-m"
api_key = "Your-API-Key"  # get it from https://dashboard.sarvam.ai/

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
).with_options(max_retries=1)

messages = [
    {"role": "system", "content": "You're a helpful AI assistant"},
    {"role": "user", "content": "Explain quantum computing in simple terms"},
]

response1 = client.chat.completions.create(
    model=model_name,
    messages=messages,
    reasoning_effort="medium",  # Enable thinking mode. `None` for disable.
    max_completion_tokens=4096,
)
print("First response:", response1.choices[0].message.content)

# Building messages for the second turn (using previous response as context)
messages.extend(
    [
        {
            "role": "assistant",
            "content": response1.choices[0].message.content,
        },
        {"role": "user", "content": "Can you give an analogy for superposition?"},
    ]
)

response2 = client.chat.completions.create(
    model=model_name,
    messages=messages,
    reasoning_effort="medium",
    max_completion_tokens=8192,
)
print("Follow-up response:", response2.choices[0].message.content)
```

Refer to API docs here: [sarvam Chat Completions API docs](https://docs.sarvam.ai/api-reference-docs/chat/completions)

`reasoning_effort` can take three possible values: `low`, `medium`, and `high` to be consistent with the OpenAI API spec. Setting any of the three values just enables the thinking mode of sarvam-m.

# VLLM Deployment

For easy deployment, we can use `vllm>=0.8.5` and create an OpenAI-compatible API endpoint with `vllm serve sarvamai/sarvam-m`.

If you want to use vLLM with python, you can do the following.

```python
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

messages = [{"role": "user", "content": "Why is 42 the best number?"}]

# By default, thinking mode is enabled.
# If you want to disable thinking, add:
# extra_body={"chat_template_kwargs": {"enable_thinking": False}}
response = client.chat.completions.create(model=model, messages=messages)
output_text = response.choices[0].message.content

if "</think>" in output_text:
    reasoning_content = output_text.split("</think>")[0].rstrip("\n")
    content = output_text.split("</think>")[-1].lstrip("\n")
else:
    reasoning_content = ""
    content = output_text

print("reasoning content:", reasoning_content)
print("content:", content)

# For the next round, add the model's response directly as assistant turn.
messages.append(
    {"role": "assistant", "content": output_text}
)
```