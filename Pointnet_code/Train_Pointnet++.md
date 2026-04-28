# Overview
This project focuses on semantic segmentation of 3D environments using point cloud data.
The goal is to identify structural elements (e.g., walls, floors, doors) to assist firefighter navigation and scene understanding.

# Setup Environment: 
pip install numpy torch

# Project Structure
├── dataset.py       # Block sampling
├── model.py         # PointNet++ model
├── parse_file.py    # Parse semantic .obj files
├── preprocess.py    # Convert mesh to point cloud
├── train.py         # Training pipeline


# Pipeline Overview
semantic.obj(from Stanford 2D-3D-Semantics Dataset)
    ↓ 
parse_file.py
    ↓ 
mesh (vertices, faces, labels)
    ↓ 
preprocess.py
    ↓ 
points.npy + labels.npy
    ↓ 
dataset.py
    ↓ 
random blocks
    ↓
model.py 
    ↓ 
train.py 
    ↓ 
best_model.pth

