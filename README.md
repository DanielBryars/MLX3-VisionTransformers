# Vision Transformers from Scratch

A from-scratch implementation of Vision Transformers (ViT) for image classification on MNIST, built with PyTorch. This project implements the complete transformer architecture including patch embedding, multi-head self-attention, and position encoding.

## Overview

This project provides a modular implementation of Vision Transformers for educational purposes, demonstrating how transformers can be applied to computer vision tasks. The implementation includes both standard and custom transformer components, with visualization tools to explore attention patterns.

## Architecture

### Core Components

- **Patch Embedder**: Divides images into patches and projects them into embedding space
- **Position Encoder**: Adds learnable positional information to patch embeddings
- **Self-Attention**: Multi-head and single-head self-attention mechanisms
- **Transformer Blocks**: Complete transformer encoder blocks with layer normalization
- **Classifier Head**: Final classification layer

### Model Structure

```
models/
├── ClassifierHead.py           # Final classification layer
├── ManualLayerNorm.py          # Custom layer normalization
├── ModelFactory.py             # Model creation and checkpoint loading
├── PatchEmbedder.py            # Image to patch embedding conversion
├── PosistionEncoder.py         # Positional encoding
├── SingleHeadSelfAttention.py  # Single attention head
├── TransformerBlock.py         # Complete transformer block
├── VisualTransformer.py        # Main ViT model
└── weights.py                  # Weight management utilities
```

## Tech Stack

- **Deep Learning**: PyTorch, torchvision
- **Experiment Tracking**: Weights & Biases (wandb)
- **Interactive Demo**: Gradio
- **Data Processing**: NumPy, pandas, datasets
- **Development**: mypy, ruff, black
- **Visualization**: matplotlib, seaborn

## Features

- From-scratch implementation of all transformer components
- Configurable architecture (patch size, embedding dimension, number of heads, etc.)
- Hyperparameter sweeps with W&B integration
- Early stopping and checkpoint management
- Interactive attention visualization with Gradio
- MNIST digit classification

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

**Single Training Run**:
```bash
python 01_training_single.py
```

This runs training with default hyperparameters:
- Patch size: 7
- Embedding size: 16
- Number of heads: 2
- Transformer blocks: 2
- Learning rate: 3e-4
- Batch size: 512

**Hyperparameter Sweep**:
```bash
python 02_training_sweep.py
```

Performs a grid search over:
- Patch sizes: [4, 7, 14]
- Embedding sizes: [8, 16, 32]
- Number of heads: [1, 2, 4]
- MLP dimensions: [32, 64, 128]
- Dropout rates: [0.0, 0.1, 0.2]

### Interactive Demo

Launch the Gradio interface to draw digits and visualize attention:

```bash
python livedemo.py [checkpoint_path]
```

If no checkpoint is provided, uses the default:
```bash
python livedemo.py
# Uses: artifacts/ts.2025_05_02__10_31_48.epoch.6.VisualTransformer.pth
```

The demo allows you to:
- Draw digits on a canvas
- See model predictions with confidence scores
- Visualize attention patterns across layers and heads

### GPU Check

Verify CUDA availability:
```bash
python gpucheck.py
```

## Project Structure

```
MLX3-VisionTransformers/
├── models/                    # Model architecture components
├── artifacts/                 # Saved checkpoints
├── docs/                      # Documentation
├── 01_training_single.py      # Single training run
├── 02_training_sweep.py       # Hyperparameter sweep
├── classifier_training.py     # Training and evaluation utilities
├── dataset.py                 # MNIST dataset loading
├── livedemo.py               # Gradio interactive demo
├── gpucheck.py               # GPU availability check
├── requirements.txt          # Python dependencies
└── README.md
```

## Hyperparameters

Key configurable parameters:

```python
hyperparameters = {
    'patch_size': 7,              # Size of image patches
    'num_classes': 10,            # Number of output classes
    'learning_rate': 3e-4,        # Adam learning rate
    'weight_decay': 0.01,         # L2 regularization
    'batch_size': 512,            # Training batch size
    'num_epochs': 10,             # Maximum epochs
    'patience': 3,                # Early stopping patience
    'num_transformer_blocks': 2,  # Number of transformer layers
    'embedding_size': 16,         # Patch embedding dimension
    'num_heads': 2,               # Multi-head attention heads
    'mlp_dim': 64,                # MLP hidden dimension
    'dropout': 0.1,               # Dropout rate
}
```

## Model Training

The training pipeline includes:

1. **Data Loading**: Automatic MNIST download and preprocessing
2. **Model Creation**: Dynamic model instantiation from hyperparameters
3. **Training Loop**: With progress bars and logging
4. **Validation**: Per-epoch validation with accuracy tracking
5. **Early Stopping**: Based on validation loss
6. **Checkpointing**: Saves best model based on validation performance
7. **W&B Logging**: All metrics logged to Weights & Biases

### Checkpoint Format

Saved checkpoints include:
```python
{
    'model_state_dict': ...,
    'hyperparameters': ...,
    'epoch': ...,
    'timestamp': ...
}
```

## Attention Visualization

The live demo provides insight into what the model learns:

- **Layer-wise Attention**: See how different layers focus on different features
- **Head-wise Patterns**: Observe specialization of attention heads
- **CLS Token Attention**: Visualize which patches contribute to classification
- **Confidence Scores**: Bar chart of per-class probabilities

## Performance

Typical results on MNIST:
- Validation Loss: ~0.20
- Validation Accuracy: ~94%
- Training time: ~2-3 minutes on GPU (10 epochs)

## Development

### Code Quality Tools

The project uses:
- **mypy**: Static type checking
- **ruff**: Fast Python linter
- **black**: Code formatting

### Reproducibility

All experiments use fixed random seeds for reproducibility:
```python
set_seed(42)
```

## Implementation Details

### Patch Embedding
Images are divided into non-overlapping patches and linearly projected:
```python
# For 28x28 MNIST with patch_size=7
# Creates 4x4 = 16 patches
# Each patch: 7x7 = 49 pixels
# Projected to embedding_size dimensions
```

### Position Encoding
Learnable 1D positional embeddings added to patch embeddings:
```python
# num_patches + 1 (for CLS token)
pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embedding_size))
```

### Multi-Head Self-Attention
Standard scaled dot-product attention with multiple heads:
```python
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

### Classification
Uses a learnable CLS token prepended to patch sequence:
```python
# CLS token aggregates information from all patches
# Final classification uses only CLS token representation
```

## Experiments

### Hyperparameter Sweep Results

The sweep explores different architectural configurations to find optimal:
- Patch granularity (4x4, 7x7, 14x14 patches)
- Model capacity (embedding size, MLP dimension)
- Attention complexity (number of heads)
- Regularization (dropout rates)

All experiments logged to W&B for comparison.

## Dataset

**MNIST**:
- Training: 60,000 images
- Testing: 10,000 images
- Image size: 28x28 grayscale
- Classes: 10 digits (0-9)

Preprocessing:
- Resize to 28x28
- Normalize: mean=0.1307, std=0.3081
- No augmentation

## Future Improvements

Potential extensions:
- Support for color images (CIFAR-10, ImageNet)
- Data augmentation
- More transformer variants (Swin, DeiT)
- Pre-training strategies
- Model distillation
- Quantization for deployment

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## License

This is an educational project for learning purposes.

## Acknowledgments

Part of the MLX (Machine Learning Experiments) series at University of Leeds MSc AI program.
