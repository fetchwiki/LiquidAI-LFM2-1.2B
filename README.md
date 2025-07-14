---
library_name: transformers
license: other
license_name: lfm1.0
license_link: LICENSE
language:
- en
- ar
- zh
- fr
- de
- ja
- ko
- es
pipeline_tag: text-generation
tags:
- liquid
- lfm2
- edge
---

<center>
<div style="text-align: center;">
  <img 
    src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/7_6D7rWrLxp2hb6OHSV1p.png" 
    alt="Liquid AI"
    style="width: 100%; max-width: 66%; height: auto; display: inline-block; margin-bottom: 0.5em; margin-top: 0.5em;"
  />
</div>

<a href="https://playground.liquid.ai/chat">
<svg width="114.8" height="20" viewBox="0 0 1300 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Liquid Playground" style="margin-bottom: 1em;">
  <title>Liquid: Playground</title>
  <g>
    <rect fill="#fff" width="600" height="200"></rect>
    <rect fill="url(#x)" x="600" width="700" height="200"></rect>
  </g>
  <g transform="translate(20, 30) scale(0.4, 0.4)">
    <path d="M172.314 129.313L172.219 129.367L206.125 188.18C210.671 195.154 213.324 203.457 213.324 212.382C213.324 220.834 210.956 228.739 206.839 235.479L275.924 213.178L167.853 33.6L141.827 76.9614L172.314 129.313Z" fill="black"/>
    <path d="M114.217 302.4L168.492 257.003C168.447 257.003 168.397 257.003 168.352 257.003C143.515 257.003 123.385 237.027 123.385 212.387C123.385 203.487 126.023 195.204 130.55 188.24L162.621 132.503L135.966 86.7327L60.0762 213.183L114.127 302.4H114.217Z" fill="black"/>
    <path d="M191.435 250.681C191.435 250.681 191.43 250.681 191.425 250.686L129.71 302.4H221.294L267.71 226.593L191.435 250.686V250.681Z" fill="black"/>
  </g>
  <g aria-hidden="true" fill="#fff" text-anchor="start" font-family="Verdana,DejaVu Sans,sans-serif" font-size="110">
    <text x="200" y="148" textLength="329" fill="#000" opacity="0.1">Liquid</text>
    <text x="190" y="138" textLength="329" fill="#000">Liquid</text>
    <text x="655" y="148" textLength="619" fill="#000" opacity="0.1">Playground</text>
    <text x="645" y="138" textLength="619">Playground</text>
  </g>
  
  <linearGradient id="x" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%" style="stop-color:#000000"></stop>
    <stop offset="100%" style="stop-color:#000000"></stop>
  </linearGradient>
</svg>
</a>
</center>

# LFM2-1.2B

LFM2 is a new generation of hybrid models developed by [Liquid AI](https://www.liquid.ai/), specifically designed for edge AI and on-device deployment. It sets a new standard in terms of quality, speed, and memory efficiency. 

We're releasing the weights of three post-trained checkpoints with 350M, 700M, and 1.2B parameters. They provide the following key features to create AI-powered edge applications:

* **Fast training & inference** – LFM2 achieves 3x faster training compared to its previous generation. It also benefits from 2x faster decode and prefill speed on CPU compared to Qwen3.
* **Best performance** – LFM2 outperforms similarly-sized models across multiple benchmark categories, including knowledge, mathematics, instruction following, and multilingual capabilities.
* **New architecture** – LFM2 is a new hybrid Liquid model with multiplicative gates and short convolutions.
* **Flexible deployment** – LFM2 runs efficiently on CPU, GPU, and NPU hardware for flexible deployment on smartphones, laptops, or vehicles.

Find more information about LFM2 in our [blog post](https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models).

## 📄 Model details

Due to their small size, **we recommend fine-tuning LFM2 models on narrow use cases** to maximize performance. 
They are particularly suited for agentic tasks, data extraction, RAG, creative writing, and multi-turn conversations. 
However, we do not recommend using them for tasks that are knowledge-intensive or require programming skills.

| Property            | Value                         |
| ------------------- | ----------------------------- |
| **Parameters**      | 1,170,340,608                 |
| **Layers**          | 16 (10 conv + 6 attn)         |
| **Context length**  | 32,768 tokens                 |
| **Vocabulary size** | 65,536                        |
| **Precision**       | bfloat16                      |
| **Training budget** | 10 trillion tokens            |
| **License**         | LFM Open License v1.0         |

**Supported languages**: English, Arabic, Chinese, French, German, Japanese, Korean, and Spanish.

**Generation parameters**: We recommend the following parameters:
* `temperature=0.3`
* `min_p=0.15`
* `repetition_penalty=1.05`

**Chat template**: LFM2 uses a ChatML-like chat template as follows:

```
<|startoftext|><|im_start|>system
You are a helpful assistant trained by Liquid AI.<|im_end|>
<|im_start|>user
What is C. elegans?<|im_end|>
<|im_start|>assistant
It's a tiny nematode that lives in temperate soil environments.<|im_end|>
```

You can apply it using the dedicated [`.apply_chat_template()`](https://huggingface.co/docs/transformers/en/chat_templating#applychattemplate) function from Hugging Face transformers.

**Tool use**: It consists of four main steps:
1. **Function definition**: LFM2 takes JSON function definitions as input (JSON objects between `<|tool_list_start|>` and `<|tool_list_end|>` special tokens), usually in the system prompt
2. **Function call**: LFM2 writes Pythonic function calls (a Python list between `<|tool_call_start|>` and `<|tool_call_end|>` special tokens), as the assistant answer.
3. **Function execution**: The function call is executed and the result is returned (string between `<|tool_response_start|>` and `<|tool_response_end|>` special tokens), as a "tool" role.
4. **Final answer**: LFM2 interprets the outcome of the function call to address the original user prompt in plain text.

Here is a simple example of a conversation using tool use:

```
<|startoftext|><|im_start|>system
List of tools: <|tool_list_start|>[{"name": "get_candidate_status", "description": "Retrieves the current status of a candidate in the recruitment process", "parameters": {"type": "object", "properties": {"candidate_id": {"type": "string", "description": "Unique identifier for the candidate"}}, "required": ["candidate_id"]}}]<|tool_list_end|><|im_end|>
<|im_start|>user
What is the current status of candidate ID 12345?<|im_end|>
<|im_start|>assistant
<|tool_call_start|>[get_candidate_status(candidate_id="12345")]<|tool_call_end|>Checking the current status of candidate ID 12345.<|im_end|>
<|im_start|>tool
<|tool_response_start|>{"candidate_id": "12345", "status": "Interview Scheduled", "position": "Clinical Research Associate", "date": "2023-11-20"}<|tool_response_end|><|im_end|>
<|im_start|>assistant
The candidate with ID 12345 is currently in the "Interview Scheduled" stage for the position of Clinical Research Associate, with an interview date set for 2023-11-20.<|im_end|>
```

**Architecture**: Hybrid model with multiplicative gates and short convolutions: 10 double-gated short-range LIV convolution blocks and 6 grouped query attention (GQA) blocks.

**Pre-training mixture**: Approximately 75% English, 20% multilingual, and 5% code data sourced from the web and licensed materials.

**Training approach**:
* Knowledge distillation using [LFM1-7B](https://www.liquid.ai/blog/introducing-lfm-7b-setting-new-standards-for-efficient-language-models) as teacher model
* Very large-scale SFT on 50% downstream tasks, 50% general domains
* Custom DPO with length normalization and semi-online datasets
* Iterative model merging

## 🏃 How to run LFM2

You can run LFM2 with transformers and llama.cpp. vLLM support is coming.

### 1. Transformers

To run LFM2, you need to install Hugging Face [`transformers`](https://github.com/huggingface/transformers) from source (v4.54.0.dev0).
You can update or install it with the following command: `pip install "transformers @ git+https://github.com/huggingface/transformers.git@main"`.

Here is an example of how to generate an answer with transformers in Python:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_id = "LiquidAI/LFM2-1.2B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="bfloat16",
    trust_remote_code=True,
#    attn_implementation="flash_attention_2" <- uncomment on compatible GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Generate answer
prompt = "What is C. elegans?"
input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True,
).to(model.device)

output = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.3,
    min_p=0.15,
    repetition_penalty=1.05,
    max_new_tokens=512,
)

print(tokenizer.decode(output[0], skip_special_tokens=False))

# <|startoftext|><|im_start|>user
# What is C. elegans?<|im_end|>
# <|im_start|>assistant
# C. elegans, also known as Caenorhabditis elegans, is a small, free-living
# nematode worm (roundworm) that belongs to the phylum Nematoda.
```

You can directly run and test the model with this [Colab notebook](https://colab.research.google.com/drive/1_q3jQ6LtyiuPzFZv7Vw8xSfPU5FwkKZY?usp=sharing).

### 2. Llama.cpp

You can run LFM2 with llama.cpp using its [GGUF checkpoint](https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF). Find more information in the model card.

## 🔧 How to fine-tune LFM2

We recommend fine-tuning LFM2 models on your use cases to maximize performance.

| Notebook | Description | Link |
|-------|------|------|
| SFT (Axolotl) | Supervised Fine-Tuning (SFT) notebook with a LoRA adapter in Axolotl. | <a href="https://colab.research.google.com/drive/155lr5-uYsOJmZfO6_QZPjbs8hA_v8S7t?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| SFT (TRL) | Supervised Fine-Tuning (SFT) notebook with a LoRA adapter in TRL. | <a href="https://colab.research.google.com/drive/1j5Hk_SyBb2soUsuhU0eIEA9GwLNRnElF?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |
| DPO (TRL) | Preference alignment with Direct Preference Optimization (DPO) in TRL. | <a href="https://colab.research.google.com/drive/1MQdsPxFHeZweGsNx4RH7Ia8lG8PiGE1t?usp=sharing"><img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/vlOyMEjwHa_b_LXysEu2E.png" width="110" alt="Colab link"></a> |

## 📈 Performance

LFM2 outperforms similar-sized models across different evaluation categories.

### 1. Automated benchmarks

![image/png](https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/3cB7VqMnrG9I8EqrL7k-q.png)

| Model | MMLU | GPQA | IFEval | IFBench | GSM8K | MGSM | MMMLU |
|-------|------|------|--------|---------|-------|------|-------|
| LFM2-350M | 43.43 | 27.46 | 65.12 | 16.41 | 30.1 | 29.52 | 37.99 |
| LFM2-700M | 49.9 | 28.48 | 72.23 | 20.56 | 46.4 | 45.36 | 43.28 |
| LFM2-1.2B | *55.23* | **31.47** | **74.89** | *20.7* | *58.3* | *55.04* | **46.73** |
| Qwen3-0.6B | 44.93 | 22.14 | 64.24 | 19.75 | 36.47 | 41.28 | 30.84 |
| Qwen3-1.7B | **59.11** | 27.72 | *73.98* | **21.27** | 51.4 | **66.56** | *46.51* |
| Llama-3.2-1B-Instruct | 46.6 | *28.84* | 52.39 | 16.86 | 35.71 | 29.12 | 38.15 |
| gemma-3-1b-it | 40.08 | 21.07 | 62.9 | 17.72 | **59.59** | 43.6 | 34.43 |

### 2. LLM-as-a-Judge

![image/png](https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/4Yxx0l9aQ6ATrps5GWHzv.png)
![image/png](https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/lzpZOGwH-8bTlOWd3tv6M.png)

### 3. Inference

#### Throughput comparison on CPU in ExecuTorch

![image/png](https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/KoKcsXUOnkvz2dwZ99k08.png)

#### Throughput comparison on CPU in Llama.cpp

![image/png](https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/c7UYZ5nh6qJMB4rd6WKde.png)

## 📬 Contact

If you are interested in custom solutions with edge deployment, please contact [our sales team](https://www.liquid.ai/contact).