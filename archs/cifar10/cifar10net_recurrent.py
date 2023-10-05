import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, layer, surrogate, neuron


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class cifar10net(nn.Module):

    def __init__(self, num_classes=10):
        super(cifar10net, self).__init__()
        self.total_timestep = 5
        self.tau = 4./3.

        self.last_back1 = 0.
        self.last_back2 = 0.

        self. rev_conv1 =  nn.Sequential(
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=self.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.rev_conv2 = nn.Sequential(
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=self.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )


        self.stem = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256))

        self.block1 = nn.Sequential(

            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=self.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256))

        self.block2 = nn.Sequential(

            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=self.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256))

        self.block3 = nn.Sequential(

            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=self.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256))


        self.maxpool1 =  nn.AvgPool2d(2, 2)

        self.block4 = nn.Sequential(

            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=self.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256))

        self.block5 = nn.Sequential(

            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=self.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256)
        )


        self.block6 = nn.Sequential(

            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=self.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256))

        self.maxpool2 = nn.Sequential(
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=self.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),

            nn.AvgPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            layer.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 128 * 4 * 4),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=self.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),
            nn.Linear(2048, 100),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=self.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),
        )
        self.boost = nn.AvgPool1d(10, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a =2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, input):

        acc_voltage = 0
        static_x = self.stem(input)

        self.last_back1 = 0.
        self.last_back2 = 0.

        for t in range(self.total_timestep):
            x1 = self.block1(static_x + self.last_back1)
            x2 = self.block2(x1)
            self.last_back1 = self.rev_conv1(x2)
            x3 = self.block3(x2)

            x4 = self.maxpool1(x3)
            x5 = self.block4(x4 + self.last_back2)
            x6 = self.block5(x5)
            self.last_back2 = self.rev_conv2(x6)
            x7 = self.block6(x6)

            x7 = self.maxpool2(x7)

            x = torch.flatten(x7, 1)
            x = self.classifier(x)

            boosted_prob = self.boost(x.unsqueeze(1)).squeeze(1)

            acc_voltage = acc_voltage + boosted_prob

        acc_voltage = acc_voltage / self.total_timestep

            # print (acc_voltage[0,...])
        return acc_voltage

    # def neuron_init(self):
    #
    #     neuron_type = 'LIFNode'
    #     for name, module in self.features.named_modules():
    #         if neuron_type in str(type(module)):
    #             module.v = 0.
    #         if 'Dropout' in str(type(module)):
    #             module.mask = None
    #
    #     for name, module in self.classifier.named_modules():
    #         if neuron_type in str(type(module)):
    #             module.v = 0.
    #         if 'Dropout' in str(type(module)):
    #             module.mask = None
    #
