import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (3, 3), stride=1)
        self.conv2 = nn.Conv2d(16, 16, (3, 3))
        self.conv3 = nn.Conv2d(16, 16, (3, 3))
        self.conv4 = nn.Conv2d(16, 32, (3, 3))
        self.conv5 = nn.Conv2d(32, 32, (3, 3))
        self.conv6 = nn.Conv2d(32, 32, (3, 3))
        self.conv7 = nn.Conv2d(32, 64, (3, 3))
        self.conv8 = nn.Conv2d(64, 64, (3, 3))
        self.conv9 = nn.Conv2d(64, 64, (3, 3))
        self.conv10 = nn.Conv2d(64, 128, (3, 3))
        self.conv11 = nn.Conv2d(128, 128, (3, 3))
        self.conv12 = nn.Conv2d(128, 256, (3, 3))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(p=0.3)
        self.dense2 = nn.Linear(64, 43)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.global_pool(x)
        x = F.relu(self.dense1(x.squeeze()))
        # TODO: add dropout
        x = self.dense2(self.dropout(x))

        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


class BasicBlock(nn.Module):
    def __init__(self, in_maps, out_maps, downsample=False):
        super(BasicBlock, self).__init__()
        self.in_maps = in_maps
        self.out_maps = out_maps
        self.conv1 = nn.Conv2d(in_maps, out_maps, (3, 3), stride=1 if not downsample else 2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_maps)
        self.conv2 = nn.Conv2d(out_maps, out_maps, (3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_maps)
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_maps, out_maps, (1, 1), stride=2),
                nn.BatchNorm2d(out_maps)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity

        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, (7, 7), stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d((3, 3), stride=1, padding=1)

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, downsample=True),
            BasicBlock(128, 128),
            BasicBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, downsample=True),
            BasicBlock(256, 256),
            BasicBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, downsample=True),
            BasicBlock(512, 512),
            BasicBlock(512, 512)
        )
        self.flatten = Flatten()
        self.dense1 = nn.Linear(4*4*512, 512)
        self.dropout = nn.Dropout(p=0.1)
        self.dense2 = nn.Linear(512, 43)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.flatten(out)
        out = F.relu(self.dense1(out))
        out = self.dropout(out)
        out = self.dense2(out)

        return out


if __name__ == '__main__':
    a = torch.randn(10, 50, 2)
    print(a.shape)
    l = Flatten()
    print("flatten a shape: ", l(a).shape)
