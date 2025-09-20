# Meat_Learning
# CNN Project for Beef Part Classification

This project implements a deep learning pipeline to classify images of beef cuts into four categories using Convolutional Neural Networks (CNNs) with PyTorch.  
Supported architectures: **VGG16**, **ResNet50**, **EfficientNet-B0**, and **InceptionV3**.

---

## 📌 Overview
- **Objective:** Classify beef parts: ribeye, chuck, tenderloin, beef_tongue
- **Framework:** PyTorch
- **Configurable:** Change model type, hyperparameters, dataset paths via YAML
- **Reusable Modules:** Training, evaluation, prediction, saving/loading models

---

## 📂 Project Structure (Bullet Format)

- **data/**
  - **raw/** – Original dataset (unprocessed)
    - **ribeye/** – Raw ribeye images
    - **chuck/** – Raw chuck images
    - **tenderloin/** – Raw tenderloin images
    - **beef_tongue/** – Raw beef tongue images
  - **processed/** – Processed dataset (ready for model)
    - **train/** – Training images
    - **val/** – Validation images

- **notebooks/**
  - `experiment_cnn.ipynb` – Jupyter Notebook for experiments

- **experiments/** – Stores experiment metadata/results  
- **logs/** – Training and validation logs  
- **saved_models/** – Trained model files (.pth)

- **src/**
  - **configs/** – Model configuration files (YAML)
    - `vgg.yaml` – Config for VGG16
    - `resnet.yaml` – Config for ResNet50
    - `efficientnet.yaml` – Config for EfficientNet-B0
    - `inception.yaml` – Config for InceptionV3
  - **models/** – Model architecture definitions
    - `vgg.py`, `resnet.py`, `efficientnet.py`, `inception.py`
  - `data_loader.py` – Loads and preprocesses datasets
  - `train.py` – Training loop  
  - `evaluate.py` – Evaluate model performance  
  - `predict.py` – Predict a single image  
  - `utils.py` – Utility functions (save/load models)

- `main.py` – Main script to train/evaluate  
- `requirements.txt` – Dependencies  
- `README.md` – Project documentation  

---

## ⚙️ How It Works

1. **Prepare the Dataset**
   - Place original images into `data/raw/<category>/`
   - Split datasets into `train` and `val` under `data/processed/`

2. **Update Configuration**
   - Edit YAML files in `src/configs/` to select model, change image size, batch size, epochs, etc.

3. **Training**
   ```bash
   python main.py