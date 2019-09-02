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


# ################ ResNet part ###################


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


# ######################## DenseNet part #################################


class DenseLayer(nn.Module):
    def __init__(self, in_features, growth_rate, bn_size, device):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_features).to(device)
        self.conv1 = nn.Conv2d(in_features, bn_size*growth_rate, (1, 1), stride=1, bias=False).to(device)

        self.bn2 = nn.BatchNorm2d(bn_size*growth_rate).to(device)
        self.conv2 = nn.Conv2d(bn_size*growth_rate, growth_rate, (3, 3), stride=1, padding=1, bias=False).to(device)

    def forward(self, *x):
        # x is list of concatenated tensors
        concat_features = torch.cat(x, dim=1)
        out = F.relu(self.bn1(concat_features))
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        return out


class Transition(nn.Module):
    def __init__(self, in_features, out_features, device):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_features).to(device)
        self.conv = nn.Conv2d(in_features, out_features, (1, 1), stride=1, bias=False).to(device)
        self.avg_pool = nn.AvgPool2d((2, 2), stride=2).to(device)

    def forward(self, x):
        out = self.avg_pool(self.conv(F.relu(self.bn(x))))

        return out


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_features, bn_size, growth_rate, device):
        super(DenseBlock, self).__init__()
        self.layers = []
        for i in range(num_layers):
            layer = DenseLayer(in_features + i*growth_rate, growth_rate, bn_size, device)
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(*features)
            features.append(out)

        return torch.cat(features, dim=1)


class DenseNet(nn.Module):
    def __init__(self, layers_config, bn_size, growth_rate, device='cpu', compression=0.5):
        super(DenseNet, self).__init__()

        in_features = 64
        self.conv1 = nn.Conv2d(3, in_features, kernel_size=(7, 7), stride=2, padding=3).to(device)
        self.bn1 = nn.BatchNorm2d(in_features).to(device)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1).to(device)

        self.layers = []
        for num_layers in layers_config[:-1]:
            print("Number of features is", in_features)
            self.layers.append(DenseBlock(num_layers, in_features, bn_size, growth_rate, device))
            in_features += num_layers * growth_rate
            self.layers.append(Transition(in_features, int(in_features*compression), device))
            in_features = int(in_features*compression)

        print("Number of features is", in_features)

        self.layers.append(DenseBlock(layers_config[-1], in_features, bn_size, growth_rate, device))
        self.layers.append(Flatten())
        print("Number of features is", in_features + layers_config[-1]*growth_rate)
        self.layers.append(nn.Linear(3*3*(in_features + layers_config[-1]*growth_rate), 200).to(device))
        self.layers.append(nn.ReLU().to(device))
        self.layers.append(nn.Linear(200, 43).to(device))
        self.layer = nn.Sequential(*self.layers).to(device)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.pool1(out)
        out = self.layer(out)

        return out


if __name__ == '__main__':
    a = torch.randn(10, 50, 2)
    print(a.shape)
    l = Flatten()
    print("flatten a shape: ", l(a).shape)
