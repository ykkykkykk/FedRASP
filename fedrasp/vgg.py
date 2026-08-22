"""VGG-16 used by the FedRASP CIFAR-10 example."""

import torch
from torch import nn

FEDRASP_VGG16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512]


class FedRASPVGG(nn.Module):
    def __init__(self, features, num_classes=10, scale=1.0):
        super().__init__()
        self.features = features
        with torch.no_grad():
            self.features.eval()
            output = self.features(torch.zeros(2, 3, 32, 32))
            flatten_dim = int(output.view(output.size(0), -1).size(1))
        hidden_dim = int(4096 * scale)
        self.projector = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        representation = self.features(inputs)
        representation = representation.view(representation.size(0), -1)
        representation = self.projector(representation)
        return {"representation": representation, "output": self.classifier(representation)}


def fedrasp_make_vgg_layers(slim_idx=0, scale=1.0, track_running_stats=True):
    layers = []
    input_channels = 3
    convolution_index = 0
    for item in FEDRASP_VGG16 + ["M"]:
        if item == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        output_channels = int(item)
        if convolution_index >= slim_idx:
            output_channels = int(output_channels * scale)
        layers.extend([
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True),
        ])
        input_channels = output_channels
        convolution_index += 1
    return nn.Sequential(*layers)


def fedrasp_vgg16_bn(num_classes=10, track_running_stats=True, slim_idx=0, scale=1.0):
    return FedRASPVGG(
        fedrasp_make_vgg_layers(slim_idx, scale, track_running_stats),
        num_classes=num_classes,
        scale=scale,
    )

