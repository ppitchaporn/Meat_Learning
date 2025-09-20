import torch.nn as nn
import torchvision.models as models

def build_efficientnet(num_classes=4, pretrained=True):
    model = models.efficientnet_b0(pretrained=pretrained)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
