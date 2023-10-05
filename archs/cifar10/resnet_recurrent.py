'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import functional, layer, surrogate, neuron


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.lif1 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau= 4./3.,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True)

        self.lif2 = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=4. / 3.,
                                   surrogate_function=surrogate.ATan(),
                                   detach_reset=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.lif1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.lif2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)


        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, total_timestep =6):
        super(ResNet, self).__init__()
        self.in_planes = 128
        self.total_timestep = total_timestep

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.lif_input = neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=4. / 3.,
                                        surrogate_function=surrogate.ATan(),
                                        detach_reset=True)

        self.layer1 = self._make_layer(block, 128, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 256, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 512, num_blocks[2], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.last_back1 = 0.
        self.last_back2 = 0.
        self.last_back3 = 0.


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        # acc_voltage = 0
        output_list = []

        static_x = self.bn1(self.conv1(x))

        for t in range(self.total_timestep):
            out1= self.lif_input(static_x)
            out2 = self.layer1(out1)
            out3 = self.layer2(out2)
            out4 = self.layer3(out3)
            out5 = F.avg_pool2d(out4, 4)
            out5 = out5.view(out5.size(0), -1)
            out6 = self.linear(out5)
            # acc_voltage = acc_voltage + out
            output_list.append(out6)

        # acc_voltage = acc_voltage / self.total_timestep

        return output_list


def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet19(num_classes, total_timestep):
    return ResNet(BasicBlock, [3,3,2], num_classes, total_timestep)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()