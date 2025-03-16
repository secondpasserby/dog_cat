import torch
import torch.nn as nn
import torchvision.models as models

class vgg11(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
   
        super().__init__()


        self.model = models.vgg11(pretrained=pretrained)

        in_features = self.model.classifier[6].in_features  # 获取全连接层输入维度
        self.model.classifier[6] = nn.Linear(in_features, num_classes)  # 替换为新的分类层

    def forward(self, x):
        return self.model(x)