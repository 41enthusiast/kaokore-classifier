import torch.nn as nn
import torch

from efficientnet_pytorch import EfficientNet
from torchvision.models import vgg16, mobilenet_v2, densenet121, resnet50


class FinetunedModel(nn.Module):
    class Identity(nn.Module):
        def __init__(self): super().__init__()

        def forward(self, x):
            return x

    def __init__(self, n_classes=2, freeze_base=True, hidden_size=512):
        super().__init__()

        self.base_model = EfficientNet.from_pretrained("efficientnet-b0")
        internal_embedding_size = self.base_model._fc.in_features
        self.base_model._fc = FinetunedModel.Identity()

        # non linear projection head improves the embedding quality
        self.fc_head = nn.Sequential(
            nn.Linear(in_features=internal_embedding_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=n_classes if n_classes > 2 else 1)
        )
        if freeze_base:
            print("Freezing embeddings")
            for param in self.base_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = self.base_model(x)
        out = self.fc_head(out)
        return out





def vgg_model(n_classes, pooling='avg', freeze_base=True):
    model = vgg16(True)
    

    if pooling=='avg':
        for i, layer in model.features.named_children():
            if isinstance(layer, torch.nn.MaxPool2d):
                model.features[int(i)] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    if freeze_base:
        print("Freezing embeddings")
        for param in model.parameters():
            param.requires_grad = False
    
    #the modified pretrained model
    model.classifier = nn.Sequential(
            nn.Linear(in_features=model.classifier[0].in_features, out_features=n_classes)
        )

    return model


def mobilenet_model(n_classes, pooling='NA', freeze_base=True):
    model = mobilenet_v2(True)

    if freeze_base:
        print("Freezing embeddings")
        for param in model.parameters():
            param.requires_grad = False
    
    #the modified pretrained model
    model.classifier[1] = nn.Sequential(
            nn.Linear(in_features=model.classifier[1].in_features, out_features=n_classes)
        )

    return model


def resnet_model(n_classes, pooling='avg', freeze_base=True):
    model = resnet50(True)

    if pooling=='avg':
        model.maxpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    if freeze_base:
        print("Freezing embeddings")
        for param in model.parameters():
            param.requires_grad = False
    
    #the modified pretrained model
    model.fc = nn.Sequential(
            nn.Linear(in_features=model.fc.in_features, out_features=n_classes)
        )

    return model


def densenet_model(n_classes, pooling='avg', freeze_base=True):
    model = densenet121(True)

    if pooling=='avg':
        for c, (i, layer) in enumerate(model.features.named_children()):
            if isinstance(layer, torch.nn.MaxPool2d):
                model.features[c] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    if freeze_base:
        print("Freezing embeddings")
        for param in model.parameters():
            param.requires_grad = False
    
    #the modified pretrained model
    model.classifier = nn.Sequential(
            nn.Linear(in_features=model.classifier.in_features, out_features=n_classes)
        )

    return model

def setup_base_models(arch, nclasses, pooling='avg'):
    models = {'vgg': vgg_model, 'resnet': resnet_model, 'mobilenet': mobilenet_model, 'densenet': densenet_model}

    return models[arch](nclasses, pooling)



