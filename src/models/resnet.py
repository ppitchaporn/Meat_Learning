import torch.nn as nn
import torchvision.models as models

def build_resnet(num_classes=4, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
