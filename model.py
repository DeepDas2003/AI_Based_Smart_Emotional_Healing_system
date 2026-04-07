import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class EmotionResNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        self.conv = nn.Conv2d(1, 64, 3, 1, 1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.layer1 = nn.Sequential(ResidualBlock(64,64), ResidualBlock(64,64))
        self.layer2 = nn.Sequential(ResidualBlock(64,128,2), ResidualBlock(128,128))
        self.layer3 = nn.Sequential(ResidualBlock(128,256,2), ResidualBlock(256,256))
        self.layer4 = nn.Sequential(ResidualBlock(256,512,2), ResidualBlock(512,512))

        self.pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,num_classes)
        )

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.fc(x)
      
