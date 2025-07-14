# LiquidAI/LFM2-1.2B

HuggingFace model: LiquidAI/LFM2-1.2B

This is a mirror of the [LiquidAI/LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B) repository from HuggingFace.

**Note**: This repository contains metadata and configuration files only. The actual model files are stored on HuggingFace due to their large size.

## Model Information

### Architecture

- **Architecture**: Lfm2ForCausalLM
- **Model Type**: lfm2
- **Vocabulary Size**: 65,536
- **Hidden Size**: 2,048
- **Number of Layers**: 16
- **Attention Heads**: 32

### Model Files

This repository contains metadata about the model files. The actual model files are stored on HuggingFace.

| File | Size | Details |
|------|------|---------|
| [model.safetensors](https://huggingface.co/LiquidAI/LFM2-1.2B/blob/main/model.safetensors) | 2GB | 148 tensors |

**Total files**: 1

### Tensor Details

Sample tensors from safetensors files:

**model.safetensors**:
    - model.embed_tokens.weight: shape=[65536,2048], dtype=BF16
    - model.embedding_norm.weight: shape=[2048], dtype=BF16
    - model.layers.0.conv.conv.weight: shape=[2048,1,3], dtype=BF16
    - model.layers.0.conv.in_proj.weight: shape=[6144,2048], dtype=BF16
    - model.layers.0.conv.out_proj.weight: shape=[2048,2048], dtype=BF16

## Usage

To use this model, visit the original HuggingFace repository:
- [LiquidAI/LFM2-1.2B](https://huggingface.co/LiquidAI/LFM2-1.2B)

## Additional Information

This mirror was created to provide easy access to model metadata and configuration files. For the actual model weights and full functionality, please visit the original repository on HuggingFace.

**Generated**: 2025-07-14
