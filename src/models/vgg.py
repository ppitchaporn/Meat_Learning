import torch.nn as nn
import torchvision.models as models

def build_vgg(num_classes=4, pretrained=True):
    model = models.vgg16(pretrained=pretrained)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model
