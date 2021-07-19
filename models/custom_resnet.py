import torch.nn as nn
import torch.nn.functional as F


class CustomResnet(nn.Module):
    def __init__(self):
        super(CustomResnet, self).__init__()
        
        # Prep Layer
        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), # 32 32  64
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Layer 1
        self.x1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), # 32 32 128
            nn.MaxPool2d(2, 2),                                                                     # 32 16 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # ResneT block
        self.res1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), # 16 16 128
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), # 16 16 128
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),  # 16 16 256
            nn.MaxPool2d(2, 2),                                                                       # 16 8 256
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Layer 3
        self.x3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False), # 8 8 512
            nn.MaxPool2d(2, 2),                                                                      # 8 4 512
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        # resnet bloack
        self.res3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False), # 4 4 512
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False),# 4 4 512
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # max pool 4D
        self.max4d = nn.MaxPool2d(4, 4)                                                             # 4 1 512
        
        # Fully connected
        self.fc = nn.Linear(in_features=512, out_features=10, bias=False)                           # 1 1 10

    def forward(self, x):
        prep_layer = self.preplayer(x)

        x1 = self.x1(prep_layer)
        R1 = self.res1(x1)
        layer1 = x1 + R1

        layer2 = self.layer2(layer1)

        x3 = self.x3(layer2)
        R3 = self.res3(x3)
        layer3 = x3 + R3

        maxpool = self.max4d(layer3)

        x = maxpool.view(maxpool.size(0), -1)
        fc = self.fc(x)
        softmax = F.log_softmax(fc.view(-1, 10), dim=-1)

        return softmax
