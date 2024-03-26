import torch.nn as nn
import torch.nn.functional as F
import torch
from Model.chanAtt import CALayer
from Model.net_02 import TemporalConvNet


class IC_BTCN(nn.Module):
    def __init__(self):
        super(IC_BTCN, self).__init__()
        self.convT1 = nn.ConvTranspose2d(in_channels=3, out_channels= 8, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=10, kernel_size=2, )
        self.ca = CALayer(channel=10)
        self.pool = nn.MaxPool2d(2, 2)
        self.TCN_N = TemporalConvNet(num_inputs=10, num_channels=[32, 22, 12, 10])
        self.TCN_R = TemporalConvNet(num_inputs=10, num_channels=[32, 22, 12, 10])
        self.fc1 = nn.Bilinear(24*10, 24*10, 32)
        self.fc2 = nn.Linear(32, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.convT1(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.ca(x))
        x = self.pool(x)
        x = torch.flatten(x, start_dim=2)
        x_1 = self.TCN_N(x)
        x = torch.flip(x, dims=[-2])
        x_2 = self.TCN_R(x)

        x = F.relu(self.fc1(torch.flatten(x_1, start_dim=1), torch.flatten(x_2, start_dim=1)))
        x = self.fc3(F.relu(self.fc2(x)))
        return x

