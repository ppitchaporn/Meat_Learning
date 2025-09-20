import yaml
import torch
from src.data_loader import get_data_loaders
from src.train import train_model
from src.evaluate import evaluate_model
from src.models.vgg import build_vgg
from src.models.resnet import build_resnet
from src.models.efficientnet import build_efficientnet
from src.models.inception import build_inception

MODEL_MAP = {
    "vgg": build_vgg,
    "resnet": build_resnet,
    "efficientnet": build_efficientnet,
    "inception": build_inception
}

if __name__ == "__main__":
    config_path = "src/configs/vgg.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_data_loaders(
        config["train_data_dir"], config["val_data_dir"],
        image_size=tuple(config["image_size"]),
        batch_size=config["batch_size"])
    model_fn = MODEL_MAP[config["model_name"]]
    model = model_fn(num_classes=config["num_classes"], pretrained=config["pretrained"])
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=config["epochs"], device=device)
    evaluate_model(model, val_loader, device=device)
