import torch.nn as nn
import torchvision.models as models

def build_inception(num_classes=4, pretrained=True):
    model = models.inception_v3(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
