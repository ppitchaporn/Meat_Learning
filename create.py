import os

# -------------------------
# Folders to create
# -------------------------
folders = [
    "data/raw/ribeye",
    "data/raw/chuck",
    "data/raw/tenderloin",
    "data/raw/beef_tongue",
    "data/processed/train",
    "data/processed/val",
    "notebooks",
    "src/configs",
    "src/models",
    "experiments",
    "logs",
    "saved_models"
]

# -------------------------
# Files with initial content
# -------------------------
files = {
    "README.md": "# CNN Project for Beef Part Classification\n\nThis project classifies images of beef cuts using multiple CNN architectures (VGG, ResNet, EfficientNet, Inception).",

    "requirements.txt": "torch\ntorchvision\npyyaml\nnumpy\nPillow\n",

    "notebooks/experiment_cnn.ipynb": "",

    # Config files
    "src/configs/vgg.yaml": 'model_name: "vgg"\nnum_classes: 4\nimage_size: [224, 224]\nbatch_size: 32\nepochs: 20\nlearning_rate: 0.001\noptimizer: "adam"\npretrained: true\ntrain_data_dir: "data/processed/train"\nval_data_dir: "data/processed/val"\n',
    "src/configs/resnet.yaml": 'model_name: "resnet"\nnum_classes: 4\nimage_size: [224, 224]\nbatch_size: 32\nepochs: 20\nlearning_rate: 0.001\noptimizer: "adam"\npretrained: true\ntrain_data_dir: "data/processed/train"\nval_data_dir: "data/processed/val"\n',
    "src/configs/efficientnet.yaml": 'model_name: "efficientnet"\nnum_classes: 4\nimage_size: [224, 224]\nbatch_size: 32\nepochs: 20\nlearning_rate: 0.001\noptimizer: "adam"\npretrained: true\ntrain_data_dir: "data/processed/train"\nval_data_dir: "data/processed/val"\n',
    "src/configs/inception.yaml": 'model_name: "inception"\nnum_classes: 4\nimage_size: [299, 299]\nbatch_size: 32\nepochs: 20\nlearning_rate: 0.001\noptimizer: "adam"\npretrained: true\ntrain_data_dir: "data/processed/train"\nval_data_dir: "data/processed/val"\n',

    # Init file
    "src/models/__init__.py": "",

    # VGG
    "src/models/vgg.py": 'import torch.nn as nn\nimport torchvision.models as models\n\ndef build_vgg(num_classes=4, pretrained=True):\n    model = models.vgg16(pretrained=pretrained)\n    model.classifier[6] = nn.Linear(4096, num_classes)\n    return model\n',

    # ResNet
    "src/models/resnet.py": 'import torch.nn as nn\nimport torchvision.models as models\n\ndef build_resnet(num_classes=4, pretrained=True):\n    model = models.resnet50(pretrained=pretrained)\n    in_features = model.fc.in_features\n    model.fc = nn.Linear(in_features, num_classes)\n    return model\n',

    # EfficientNet
    "src/models/efficientnet.py": 'import torch.nn as nn\nimport torchvision.models as models\n\ndef build_efficientnet(num_classes=4, pretrained=True):\n    model = models.efficientnet_b0(pretrained=pretrained)\n    in_features = model.classifier[1].in_features\n    model.classifier[1] = nn.Linear(in_features, num_classes)\n    return model\n',

    # Inception
    "src/models/inception.py": 'import torch.nn as nn\nimport torchvision.models as models\n\ndef build_inception(num_classes=4, pretrained=True):\n    model = models.inception_v3(pretrained=pretrained)\n    in_features = model.fc.in_features\n    model.fc = nn.Linear(in_features, num_classes)\n    return model\n',

    # Data loader
    "src/data_loader.py": 'from torchvision import datasets, transforms\nfrom torch.utils.data import DataLoader\n\ndef get_data_loaders(train_dir, val_dir, image_size=(224, 224), batch_size=32):\n    transform = transforms.Compose([\n        transforms.Resize(image_size),\n        transforms.ToTensor(),\n        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n    ])\n    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)\n    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)\n    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)\n    return train_loader, val_loader\n',

    # Train loop
    "src/train.py": 'import torch\n\ndef train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, device="cpu"):\n    model.to(device)\n    for epoch in range(epochs):\n        model.train()\n        running_loss = 0.0\n        correct, total = 0, 0\n        for inputs, labels in train_loader:\n            inputs, labels = inputs.to(device), labels.to(device)\n            optimizer.zero_grad()\n            outputs = model(inputs)\n            loss = criterion(outputs, labels)\n            loss.backward()\n            optimizer.step()\n            running_loss += loss.item()\n            _, predicted = torch.max(outputs, 1)\n            total += labels.size(0)\n            correct += (predicted == labels).sum().item()\n        acc = correct / total * 100\n        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} Accuracy: {acc:.2f}%")\n',

    # Evaluate
    "src/evaluate.py": 'import torch\n\ndef evaluate_model(model, val_loader, device="cpu"):\n    model.eval()\n    model.to(device)\n    correct, total = 0, 0\n    with torch.no_grad():\n        for inputs, labels in val_loader:\n            inputs, labels = inputs.to(device), labels.to(device)\n            outputs = model(inputs)\n            _, predicted = torch.max(outputs.data, 1)\n            total += labels.size(0)\n            correct += (predicted == labels).sum().item()\n    print(f"Validation Accuracy: {100 * correct / total:.2f}%")\n',

    # Prediction
    "src/predict.py": 'import torch\nfrom PIL import Image\nfrom torchvision import transforms\n\ndef predict_image(model, image_path, class_names, image_size=(224, 224), device="cpu"):\n    model.eval()\n    transform = transforms.Compose([\n        transforms.Resize(image_size),\n        transforms.ToTensor(),\n        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n    ])\n    image = Image.open(image_path).convert("RGB")\n    img_tensor = transform(image).unsqueeze(0).to(device)\n    with torch.no_grad():\n        outputs = model(img_tensor)\n        _, predicted = torch.max(outputs, 1)\n    return class_names[predicted.item()]\n',

    # Utils
    "src/utils.py": 'import torch\n\ndef save_model(model, path):\n    torch.save(model.state_dict(), path)\n\ndef load_model(model, path):\n    model.load_state_dict(torch.load(path))\n    model.eval()\n    return model\n',

    # Main
    "main.py": 'import yaml\nimport torch\nfrom src.data_loader import get_data_loaders\nfrom src.train import train_model\nfrom src.evaluate import evaluate_model\nfrom src.models.vgg import build_vgg\nfrom src.models.resnet import build_resnet\nfrom src.models.efficientnet import build_efficientnet\nfrom src.models.inception import build_inception\n\nMODEL_MAP = {\n    "vgg": build_vgg,\n    "resnet": build_resnet,\n    "efficientnet": build_efficientnet,\n    "inception": build_inception\n}\n\nif __name__ == "__main__":\n    config_path = "src/configs/vgg.yaml"\n    with open(config_path, "r") as f:\n        config = yaml.safe_load(f)\n    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n    train_loader, val_loader = get_data_loaders(\n        config["train_data_dir"], config["val_data_dir"],\n        image_size=tuple(config["image_size"]),\n        batch_size=config["batch_size"])\n    model_fn = MODEL_MAP[config["model_name"]]\n    model = model_fn(num_classes=config["num_classes"], pretrained=config["pretrained"])\n    criterion = torch.nn.CrossEntropyLoss()\n    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])\n    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=config["epochs"], device=device)\n    evaluate_model(model, val_loader, device=device)\n',
}

# -------------------------
# Create folders
# -------------------------
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# -------------------------
# Create files, ensuring directories exist
# -------------------------
for filepath, content in files.items():
    dirpath = os.path.dirname(filepath)
    if dirpath:  # avoid trying to create "" directory
        os.makedirs(dirpath, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ CNN multi-model project scaffold created in current directory!")