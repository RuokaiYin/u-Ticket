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
        self.total_timestep = 6
        self.tau = 4./3.

        self.features = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),

            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=self.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=self.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=self.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=self.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=self.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, tau=self.tau,
                           surrogate_function=surrogate.ATan(),
                           detach_reset=True),

            nn.MaxPool2d(2, 2)
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
        static_x = self.features[0](input)


        for t in range(self.total_timestep):
            x = self.features[1:](static_x)
            x = torch.flatten(x, 1)
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
