import torch.nn as nn
class CALayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(CALayer, self).__init__()
        self.ave_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channel, _, _ = x.size()
        y = self.ave_pool(x).view(batch, channel)
        y = self.fc(y).view(batch, channel, 1, 1)
        return x * y.expand_as(x)
