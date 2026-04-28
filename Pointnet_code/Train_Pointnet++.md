# Overview
This project focuses on semantic segmentation of 3D environments using point cloud data.
The goal is to identify structural elements (e.g., walls, floors, doors) to assist firefighter navigation and scene understanding.

# Setup Environment: 
pip install numpy torch

## Project Structure

```text
├── dataset.py       # Block sampling
├── model.py         # PointNet++ model
├── parse_file.py    # Parse semantic .obj files
├── preprocess.py    # Convert mesh to point cloud
├── train.py         # Training pipeline
```


## Pipeline Overview

```text
semantic.obj (from Stanford 2D-3D-Semantics Dataset)
    ↓
preprocess.py
    ├── uses parse_file.py to extract vertices, faces, and labels
    └── samples points from labeled mesh faces
    ↓
points.npy + labels.npy
    ↓
dataset.py
    └── samples random local blocks
    ↓
model.py
    └── PointNet++ model
    ↓
train.py
    ↓
best_model.pth
```

## Implementation Steps

### 1. Setup Environment

Make sure you have installed PyTorch and downloaded the `semantic.obj` file from the Stanford 2D-3D-Semantics Dataset.

```bash
pip install numpy torch
```

---

### 2. Convert Mesh to Point Cloud

```bash
python preprocess.py
```

This step will:

* parse the `.obj` file (via function imported from `parse_file.py`)
* extract vertices, faces, and semantic labels from obj
* sample points from mesh faces

Outputs:

```text
points.npy   # shape [N, 3]
labels.npy   # shape [N]
```

---

### 3. Train the Model

```bash
python train.py
```

During training:

* point clouds are sampled into blocks via `dataset.py`
* each block is fed into the PointNet++ model defined in `model.py`
* the model is optimized using cross-entropy loss
* the best model is saved as: best_model.pth




