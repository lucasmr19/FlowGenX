# 📦 `data_utils` Module

Utilities for **data ingestion, preprocessing, and loading** in the network traffic modeling framework.

This module is responsible for transforming raw **PCAP files** into structured data ready for machine learning pipelines, while enforcing **reproducibility**, **modularity**, and **no data leakage**. Is **the first step** in the whole `nf_framework pipeline`:

```
PCAP → Representation → Generative Model → Synthetic Traffic → Evaluation
```

## 🚀 Overview

The module implements the following pipeline:

```
PCAP → Packets → ParsedPackets → Flows / Windows → Tensors → DataLoaders
```

It is divided into two main components:

- **`preprocessing.py`** → Raw data processing (PCAP → structured samples)
- **`loaders.py`** → PyTorch datasets and dataloaders

## 🧱 Architecture

### 1. Preprocessing Pipeline (`preprocessing.py`)

Transforms raw PCAP files into structured traffic samples.

#### Pipeline stages:

1. **PCAPReader**
   - Reads packets using Scapy
   - Supports:
     - Streaming (for large files)
     - Protocol filtering (TCP, UDP, ICMP)
     - Packet limits

2. **PacketParser**
   - Converts raw packets into `ParsedPacket`
   - Extracts:
     - Ethernet, IP, TCP/UDP/ICMP fields
     - Payload (truncated)
     - Inter-arrival times (IAT)

3. **Traffic Aggregators**

   Two aggregation strategies are implemented:
   - **FlowAggregator**
     - Groups packets into **bidirectional flows** (5-tuple canonicalization)
     - Handles:
       - Timeouts
       - Flow splitting
       - Statistical feature extraction

   - **PacketWindowAggregator**
     - Groups packets into **fixed-size sequential windows**
     - Useful for sequence-based models

4. **FeatureNormalizer**
   - Normalizes numerical features:
     - `minmax`
     - `z-score`

   - Designed to avoid **data leakage** (fit only on train data)

### 2. Data Loading (`loaders.py`)

Provides PyTorch-compatible abstractions for training pipelines.

#### Core classes:

##### `TrafficDataset`

- Generic dataset over `List[TrafficSample]`
- Supports:
  - Optional transform (`TrafficSample → Any`)
  - Optional label function

##### `RepresentationDataset`

- Applies a `TrafficRepresentation` on-the-fly
- Key features:
  - Lazy encoding (`encode()` per sample)
  - Optional in-memory caching
  - Returns PyTorch tensors

##### `TrafficDataModule`

High-level orchestration component:

- Train/Val/Test splitting (reproducible via seed)
- Ensures:
  - Representation is **fit only on training data**

- Provides:
  - `train_dataloader()`
  - `val_dataloader()`
  - `test_dataloader()`

- Supports:
  - Custom batch size
  - Multi-worker loading
  - Optional caching

## 🔄 End-to-End Usage

### From PCAP to DataLoaders

```python
from data_utils.loaders import build_datamodule_from_pcap
from representations.sequential import SequentialRepresentation

# Define representation
rep = SequentialRepresentation(config)

# Build DataModule
dm = build_datamodule_from_pcap(
    "traffic.pcap",
    representation=rep,
    batch_size=64,
)

# Setup (splits + fit representation)
dm.setup()

# Training loop
for batch in dm.train_dataloader():
    print(batch.shape)
```

## ⚙️ Design Principles

### 1. Separation of Concerns

- **DataModule** handles raw data and splits
- **Representations** are applied lazily
- Avoids unnecessary memory usage

### 2. No Data Leakage

- Representations and normalizers are:
  - `fit()` → only on training data
  - `transform()` → on validation/test

### 3. Scalability

- Streaming PCAP reading
- Optional caching
- Configurable batching and workers

### 4. Flexibility

- Pluggable:
  - Aggregators
  - Representations
  - Label functions

## 📊 Data Structures

### `ParsedPacket`

Normalized representation of a network packet:

- Multi-layer fields (Ethernet, IP, TCP/UDP, ICMP)
- Payload (optional)
- Timing information (timestamp, IAT)

### `Flow`

- Bidirectional packet collection
- Includes:
  - Packet list
  - Duration
  - Statistical features

### `PacketWindow`

- Fixed-size sequence of packets
- Used for temporal models

## 🧪 Reproducibility

- Deterministic dataset splitting via `seed`
- Explicit `setup()` step
- Consistent pipeline behavior across runs

## 🏗️ Factory Helper

### `build_datamodule_from_pcap`

Convenience function that builds the full pipeline:

```python
dm = build_datamodule_from_pcap(
    pcap_path="capture.pcap",
    representation=rep,
    aggregator=FlowAggregator,
    batch_size=32,
)
```

Internally executes:

```
PCAP → PCAPPipeline → TrafficSamples → TrafficDataModule
```

## ⚠️ Notes

- `RepresentationDataset` assumes the representation is already fitted
- Use `TrafficDataModule.setup()` before accessing dataloaders
- Large PCAPs should use `streaming=True` to avoid memory issues
