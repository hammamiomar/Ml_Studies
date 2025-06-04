# ML Studies

Post Recurse Center, this is a repository documenting my deep dive into machine learning fundamentals and modern architectures. I'll probably blog some of the important insights. Not sure if im gonna upload my handwritten math/notes-- theyre really ugly. we shall see.


## Topics to be Covered

### Core Architectures & Algorithms
- **Transformers**: Attention mechanisms, multi-head attention, positional encodings
- **Vision Transformers (ViT)**: Adapting transformers for computer vision
- **Diffusion Models**: Score matching, denoising, DDPM
- **Variational Autoencoders**: ELBO derivation, reparameterization trick
- **Contrastive Learning**: CLIP, SimCLR implementations

### Probabilistic Machine Learning
- **Bayesian Neural Networks**: Uncertainty quantification
- **Variational Inference**: Mean field, stochastic VI
- **MCMC Methods**: HMC, NUTS implementations
- **Normalizing Flows**: Change of variables, coupling layers

### Optimization & Efficiency
- **Attention Optimization**: KV cache, Flash Attention concepts
- **Mixed Precision Training**: FP16/BF16, gradient scaling
- **Model Parallelism**: Tensor and pipeline parallelism
- **Distributed Training**: Data parallel, model parallel strategies

### Training Dynamics
- **Optimization Theory**: Adam, RMSProp, learning rate scheduling
- **Regularization**: Dropout, weight decay, gradient clipping
- **Loss Landscapes**: Visualization and analysis

### Mathematical Foundations
- Probability theory and Bayesian inference
- Information theory (KL divergence, mutual information)
- Linear algebra for ML (eigendecomposition, SVD)
- Calculus of variations (for understanding ELBO)

## Structure

Week by week basis. Each week will contain some algos, projects, whatever.

## Resources & References

### Primary Texts
- **Hands-On Large Language Models** - Practical transformer implementations
- **Pattern Recognition and Machine Learning (Bishop)** - Probabilistic foundations
- **Understanding Deep Learning (Prince)** - Modern architectures
- **Scaling and Parallelizing Machine Learning** - Production systems

## Setup

```bash
# Using uv for package management
uv sync
uv run python implementations/attention_basic.py
```