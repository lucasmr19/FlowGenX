# FlowGenX: Modular Network Flow Modeling & Generation Framework

**FlowGenX** is a modular and extensible framework for network traffic analysis, modeling, and generative research. Its primary focus is **generative modeling of network flows**, but it also provides tools for representation learning, synthetic data creation, and evaluation. The framework supports multiple representations and integrates with generative models such as Diffusion Models, GANs, and Transformers, making it ideal for experimentation, reproducibility, and benchmarking.

---

## Table of Contents

- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Synthetic Data Generation](#synthetic-data-generation)
  - [Representations](#representations)
  - [Generative Models](#generative-models)
  - [Training Pipeline](#training-pipeline)

- [Testing](#testing)
- [Experiments](#experiments)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

---

## Key Features

- **Data Handling**:
  - Load, preprocess, and batch network flows
  - Support for both synthetic and real-world traffic datasets
  - Flexible `TrafficDataModule` for train/validation/test splits

- **Flow Representations**:
  - Sequential tokenizers: `FlatTokenizer`, `ProtocolAwareTokenizer`
  - Image-like representations: `GASFRepresentation`, `NprintRepresentation`
  - Fully pluggable representation registry
  - Save/load support for reproducibility

- **Generative Models**:
  - Diffusion Models (DDPM with UNet backbone)
  - GANs (Generator + Discriminator)
  - Transformer-based sequence models
  - Unified API for `build()`, `train_step()`, and `generate()`

- **Multi-Purpose Pipeline**:
  - Supports generation, classification, reconstruction, and representation evaluation
  - Extensible modular design allows integration of new models and representations

- **Testing & Reproducibility**:
  - Unit and integration tests for core modules
  - Synthetic flows for fully reproducible end-to-end tests

- **Utilities**:
  - Synthetic flow generator (`utils.make_synthetic_flow`)
  - Structured logging for experiments

---

## Project Structure

```text
configs/                # Configurations for models, data, and training
data/
  loaders.py            # Data loading & batching utilities
  preprocessing.py      # Flow preprocessing tools
evaluation/             # Evaluation scripts & metrics
experiments/            # Training experiments, hyperparameter sweeps, reproducible scripts
models_ml/              # ML models implementations
  base.py
  diffusion/            # DDPM, UNet
  gan/                  # GANs
  transformer/          # Transformer-based models
representations/        # Flow representation modules
  sequential/           # Tokenizers (Flat, ProtocolAware)
  vision/               # Image-like representations (GASF, Nprint)
tests/                  # Unit & integration tests
training/               # Training managers & pipeline utilities
utils/                  # Helper functions & synthetic flow generation
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/FlowGenX.git
cd FlowGenX
```

2. Install dependencies (PyTorch >= 2.0 recommended):

```bash
pip install -r requirements.txt
```

3. Optional: install additional packages for evaluation and visualization:

```bash
pip install matplotlib scikit-image numpy pandas
```

---

## Usage

### Synthetic Data Generation

Generate synthetic flows for experimentation or testing:

```python
from utils.make_synthetic_flow import make_synthetic_flow, make_dataset

flows = make_dataset(num_flows=100)
print(f"Generated {len(flows)} synthetic flows")
```

---

### Representations

Multiple representations are supported:

```python
from representations.sequential.tokenizer import FlatTokenizer, SequentialConfig
from representations.vision.gasf import GASFRepresentation, GASFConfig
from representations.vision.nprint import NprintRepresentation, NprintConfig

# Flat tokenizer
flat = FlatTokenizer(SequentialConfig(max_length=64))
flat.fit(flows[:50])
encoded = flat.encode(flows[51])

# GASF representation
gasf = GAFRepresentation(GAFConfig(image_size=32, n_steps=32, method="s")) # "s" -> summation, "d" -> difference
gasf.fit(flows[:50])
img = gasf.encode(flows[51])

# Nprint representation
nprint = NprintRepresentation(NprintConfig(max_packets=10))
nprint.fit(flows[:50])
matrix = nprint.encode(flows[51])
```

- All representations support `.encode()`
- Some support `.decode()` for invertible representations

---

### Generative Models

Unified API for all generative models:

```python
from generative_models.diffusion.ddpm import DDPMConfig, DDPM
from generative_models.gan.model import GANConfig, GAN
from generative_models.transformer.model import TransformerConfig, TransformerModel

cfg = DDPMConfig(input_shape=(64,), timesteps=10)
model = DDPM(cfg)
model.build()  # initialize model parameters

batch = torch.randn(8, 64)
loss = model.train_step(batch)
samples = model.generate(3)
```

---

### Training Pipeline

A typical mini-training loop:

```python
from data.loaders import TrafficDataModule
from representations.sequential.tokenizer import FlatTokenizer, SequentialConfig
from generative_models.diffusion.ddpm import DDPM, DDPMConfig

flows = make_dataset(50)
rep = FlatTokenizer(SequentialConfig(max_length=64))
rep.fit(flows[:40])

dm = TrafficDataModule(flows, rep, batch_size=8)
dm.setup()
train_dl = dm.train_dataloader()

model = DDPM(DDPMConfig(input_shape=(64,), timesteps=10))
model.build()

for batch in train_dl:
    loss = model.train_step(batch)
    print("Loss:", loss.item())

samples = model.generate(5)
print("Generated samples:", samples.shape)
```

- Supports **end-to-end generation**, representation fitting, and model training
- Modular design allows swapping models or representations seamlessly

---

## Testing

Run all tests:

```bash
python -m pytest tests -v
```

- Integration tests cover all representations (Flat, ProtocolAware, GASF, Nprint)
- Full data → representation → generative model pipeline

---

## Experiments

- Place experiment scripts in `experiments/`
- Hyperparameter sweeps, model comparisons, and reproducible workflows
- Use `configs/` to store all experiment configurations

---

## Evaluation

- Evaluate generative models on:
  - Reconstruction accuracy (for invertible representations)
  - Perplexity for sequential flows
  - Image-based metrics (MSE, SSIM, FID) for GASF/Nprint

- `evaluation/` folder can host benchmarking scripts and metric calculators

---

## Contributing

- Fork the repository and follow modular structure
- Write unit and integration tests for new modules
- Submit PRs with detailed descriptions

---

## License

MIT License © 2026
