from torch.nn import Module, Sequential, MaxPool2d, AvgPool2d
from ConvBlock import ConvBlock
from torch import cat


class InceptionV1(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(InceptionV1, self).__init__()
        assert out_channels % 8 == 0, 'number of out channels must be divisible by 8.'
        basic_num = out_channels // 4
        self.branch1 = Sequential(ConvBlock(in_channels, basic_num, kernel_size=1))
        self.branch2 = Sequential(ConvBlock(in_channels, basic_num * 3, kernel_size=1),
                                  ConvBlock(basic_num * 3, basic_num * 4, kernel_size=3, padding=1))
        self.branch3 = Sequential(ConvBlock(in_channels, basic_num, kernel_size=1),
                                  ConvBlock(basic_num, basic_num, kernel_size=5, padding=2))
        self.branch4 = Sequential(MaxPool2d(kernel_size=3, stride=1, padding=1),
                                  ConvBlock(in_channels, basic_num, kernel_size=1))

    def forward(self, x):
        return cat((self.branch1(x),
                    self.branch2(x),
                    self.branch3(x),
                    self.branch4(x)
                    ), dim=1)


class InceptionV23(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(InceptionV23, self).__init__()
        assert out_channels % 16 == 0, 'number of out channels must be divisible by 16.'
        basic_num = out_channels // 8
        self.branch1 = Sequential(ConvBlock(in_channels, basic_num, kernel_size=1))
        self.branch2 = Sequential(ConvBlock(in_channels, basic_num * 3, kernel_size=1), )
        self.branch2_1 = Sequential(ConvBlock(basic_num * 3, basic_num * 2, kernel_size=(1, 3), padding=(0, 1)))
        self.branch2_2 = Sequential(ConvBlock(basic_num * 3, basic_num * 2, kernel_size=(3, 1), padding=(1, 0)))
        self.branch3 = Sequential(ConvBlock(in_channels, basic_num, kernel_size=1),
                                  ConvBlock(basic_num, basic_num // 2, kernel_size=3, padding=1))
        self.branch3_1 = Sequential(ConvBlock(basic_num // 2, basic_num // 2, kernel_size=(1, 3), padding=(0, 1)))
        self.branch3_2 = Sequential(ConvBlock(basic_num // 2, basic_num // 2, kernel_size=(3, 1), padding=(1, 0)))
        self.branch4 = Sequential(MaxPool2d(kernel_size=3, stride=1, padding=1),
                                  ConvBlock(in_channels, basic_num, kernel_size=1))

    def forward(self, x):
        b2_tmp = self.branch2(x)
        b3_tmp = self.branch3(x)
        return cat((self.branch1(x),
                    self.branch2_1(b2_tmp),
                    self.branch2_2(b2_tmp),
                    self.branch3_1(b3_tmp),
                    self.branch3_2(b3_tmp),
                    self.branch4(x)
                    ), dim=1)


class InceptionA(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(InceptionA, self).__init__()
        assert out_channels % 12 == 0, 'number of out channels must be divisible by 12.'
        basic_num = out_channels // 12
        self.branch1 = Sequential(ConvBlock(in_channels, basic_num * 3, kernel_size=1))
        self.branch2 = Sequential(ConvBlock(in_channels, basic_num * 2, kernel_size=1),
                                  ConvBlock(basic_num * 2, basic_num * 3, kernel_size=3, padding=1))
        self.branch3 = Sequential(ConvBlock(in_channels, basic_num * 2, kernel_size=1),
                                  ConvBlock(basic_num * 2, basic_num * 3, kernel_size=3, padding=1),
                                  ConvBlock(basic_num * 3, basic_num * 3, kernel_size=3, padding=1))
        self.branch4 = Sequential(AvgPool2d(kernel_size=3, stride=1, padding=1),
                                  ConvBlock(in_channels, basic_num * 3, kernel_size=1))

    def forward(self, x):
        return cat((self.branch1(x),
                    self.branch2(x),
                    self.branch3(x),
                    self.branch4(x)), dim=1)


class InceptionB(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(InceptionB, self).__init__()
        assert out_channels % 16 == 0, 'number of out channels must be divisible by 16.'
        basic_num = out_channels // 16
        self.branch1 = Sequential(ConvBlock(in_channels, basic_num * 6, kernel_size=1))
        self.branch2 = Sequential(ConvBlock(in_channels, basic_num * 2, kernel_size=1),
                                  ConvBlock(basic_num * 2, basic_num * 3, kernel_size=(1, 7), padding=(0, 3)),
                                  ConvBlock(basic_num * 3, basic_num * 4, kernel_size=(7, 1), padding=(3, 0)))
        self.branch3 = Sequential(ConvBlock(in_channels, basic_num * 2, kernel_size=1),
                                  ConvBlock(basic_num * 2, basic_num * 2, kernel_size=(1, 7), padding=(0, 3)),
                                  ConvBlock(basic_num * 2, basic_num * 3, kernel_size=(7, 1), padding=(3, 0)),
                                  ConvBlock(basic_num * 3, basic_num * 3, kernel_size=(1, 7), padding=(0, 3)),
                                  ConvBlock(basic_num * 3, basic_num * 4, kernel_size=(7, 1), padding=(3, 0)))
        self.branch4 = Sequential(AvgPool2d(kernel_size=3, stride=1, padding=1),
                                  ConvBlock(in_channels, basic_num * 2, kernel_size=1))

    def forward(self, x):
        return cat((self.branch1(x),
                    self.branch2(x),
                    self.branch3(x),
                    self.branch4(x)), dim=1)


class InceptionC(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(InceptionC, self).__init__()
        assert out_channels % 12 == 0, 'number of out channels must be divisible by 12.'
        basic_num = out_channels // 12
        self.branch1 = Sequential(ConvBlock(in_channels, basic_num * 2, kernel_size=1))
        self.branch2 = Sequential(ConvBlock(in_channels, basic_num * 2, kernel_size=1),
                                  ConvBlock(basic_num * 2, basic_num * 3, kernel_size=3))
        self.branch2_1 = Sequential(ConvBlock(basic_num * 3, basic_num * 2, kernel_size=(3, 1), padding=(1, 0)))
        self.branch2_1 = Sequential(ConvBlock(basic_num * 3, basic_num * 2, kernel_size=(1, 3), padding=(0, 1)))
        self.branch3 = Sequential(ConvBlock(in_channels, basic_num * 2, kernel_size=1),
                                  ConvBlock(basic_num * 2, basic_num * 3, kernel_size=(1, 3), padding=(0, 1)),
                                  ConvBlock(basic_num * 3, basic_num * 4, kernel_size=(3, 1), padding=(1, 0)))
        self.branch3_1 = Sequential(ConvBlock(basic_num * 4, basic_num * 2, kernel_size=(3, 1), padding=(1, 0)))
        self.branch3_1 = Sequential(ConvBlock(basic_num * 4, basic_num * 2, kernel_size=(1, 3), padding=(0, 1)))
        self.branch4 = Sequential(AvgPool2d(kernel_size=3, stride=1, padding=1),
                                  ConvBlock(in_channels, basic_num * 2, kernel_size=1))

    def forward(self, x):
        b2_tmp = self.branch2(x)
        b3_tmp = self.branch3(x)
        return cat((self.branch1(x),
                    self.branch2_1(b2_tmp),
                    self.branch2_2(b2_tmp),
                    self.branch3_1(b3_tmp),
                    self.branch3_2(b3_tmp),
                    self.branch4(x)
                    ), dim=1)


class ReductionA(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ReductionA, self).__init__()
        assert (out_channels - in_channels) % 2 == 0, 'output should correspond to input.'
        basic_num = (out_channels - in_channels) // 2
        self.branch1 = Sequential(ConvBlock(in_channels, basic_num, kernel_size=3, stride=2))
        self.branch2 = Sequential(ConvBlock(in_channels, in_channels, kernel_size=1),
                                  ConvBlock(in_channels, in_channels, kernel_size=3, padding=1),
                                  ConvBlock(in_channels, basic_num, kernel_size=3, stride=2))
        self.branch3 = Sequential(MaxPool2d(kernel_size=3, stride=2, padding=0))

    def forward(self, x):
        return cat((self.branch1(x),
                    self.branch2(x),
                    self.branch3(x)
                    ), dim=1)


class ReductionB(Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ReductionB, self).__init__()
        assert (out_channels - in_channels) % 2 == 0, 'output should correspond to input.'
        if in_channels > 192:
            advise = (192, 256, 320)
        else:
            advise = (in_channels for x in range(3))
        self.branch1 = Sequential(ConvBlock(in_channels, advise[0], kernel_size=1),
                                  ConvBlock(advise[0], advise[0], kernel_size=3, stride=2))
        self.branch2 = Sequential(ConvBlock(in_channels, advise[1], kernel_size=1),
                                  ConvBlock(advise[1], advise[1], kernel_size=(1, 7), padding=(0, 3)),
                                  ConvBlock(advise[1], advise[1], kernel_size=(7, 1), padding=(3, 0)),
                                  ConvBlock(advise[1], advise[2], kernel_size=3, stride=2))
        self.branch3 = Sequential(MaxPool2d(kernel_size=3, stride=2, padding=0))

    def forward(self, x):
        return cat((self.branch1(x),
                    self.branch2(x),
                    self.branch3(x)
                    ), dim=1)
