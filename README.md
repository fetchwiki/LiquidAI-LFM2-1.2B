# LiquidAI/LFM2-1.2B

HuggingFace model: LiquidAI/LFM2-1.2B

This is a mirror of the [LiquidAI/LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B) repository from HuggingFace.

**Note**: This repository contains metadata and configuration files only. The actual model files are stored on HuggingFace due to their large size.

## Repository Overview

| Type | Count | Total Size |
|------|-------|------------|
| Model Files | 1 | 2.18GB |
| Config Files | 5 | 4.6MB |
| Code Files | 0 | 0B |
| Documentation | 2 | 22.39KB |

## Model Information

### Architecture Details

- **Architecture**: Lfm2ForCausalLM
- **Model Type**: lfm2
- **Vocabulary Size**: 65,536
- **Hidden Size**: 2,048
- **Number of Layers**: 16
- **Attention Heads**: 32
- **Max Position Embeddings**: 128,000

## Model Files

This repository contains metadata about the model files. The actual model files are stored on HuggingFace.

| File | Size | Type | Details |
|------|------|------|----------|
| [model.safetensors](https://huggingface.co/LiquidAI/LFM2-1.2B/blob/main/model.safetensors) | 2GB | safetensors | 148 tensors |

**Total model files**: 1

## Tensor Analysis

### model.safetensors

**Tensor Overview:**
- Total tensors: 148
- Total parameters: 1,170,340,608
- Data types: BF16 (148)
- Top layers: model (148)

**Sample Tensors:**
```
    - model.embed_tokens.weight: shape=[65536,2048], dtype=BF16, params=134,217,728
    - model.embedding_norm.weight: shape=[2048], dtype=BF16, params=2,048
    - model.layers.0.conv.conv.weight: shape=[2048,1,3], dtype=BF16, params=6,144
    - model.layers.0.conv.in_proj.weight: shape=[6144,2048], dtype=BF16, params=12,582,912
    - model.layers.0.conv.out_proj.weight: shape=[2048,2048], dtype=BF16, params=4,194,304
```

## Configuration Files

| File | Category | Size |
|------|----------|------|
| config.json | model | 1000B |
| generation_config.json | generation | 137B |
| special_tokens_map.json | other | 434B |
| tokenizer.json | tokenizer | 4MB |
| tokenizer_config.json | tokenizer | 89KB |

## Usage

To use this model, visit the original HuggingFace repository:
- [LiquidAI/LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B)

## Additional Information

This mirror was created to provide easy access to model metadata and configuration files. For the actual model weights and full functionality, please visit the original repository on HuggingFace.

---

**Repository Statistics:**
- Total files analyzed: 10
- Total size: 2.18GB
- Model files: 1
- Generated: 2025-07-14
