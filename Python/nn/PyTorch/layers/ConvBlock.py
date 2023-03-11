from torch.nn import Module, Conv2d, BatchNorm2d
from torch.nn.functional import relu


class ConvBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = relu(x)
        return x
