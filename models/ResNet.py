import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.model = models.resnet18(pretrained=False)  # 加载 ResNet18
        self.model.fc = nn.Linear(512, num_classes)  # 修改最后一层

    def forward(self, x):
        return self.model(x)