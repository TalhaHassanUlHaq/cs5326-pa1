"""Source package for CS 5326 Programming Assignment 1: The Modern Transformer LM.

Modules:
    - layers: Linear, Embedding, RMSNorm, SiLU, SwiGLU
    - rope: RotaryPositionalEmbedding (adjacent-pair convention)
    - attention: Softmax, ScaledDotProductAttention, CausalGroupedQueryAttention
    - model: TransformerBlock, TransformerLM
    - optim: CrossEntropy, AdamW, Cosine LR Schedule, Gradient Clipping
    - data: Memmap data loader (uint16) and batch sampling
    - checkpoint: Checkpoint serialization and restoration
    - generate: Autoregressive decoding with temperature & nucleus sampling
    - train: Training loop script
"""
