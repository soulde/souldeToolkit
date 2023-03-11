from torch.nn import Module, Conv2d, MaxPool2d, Linear
from torch.nn.functional import relu


class LeNet(Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = Conv2d(3, 16, 5)
        self.pool1 = MaxPool2d(2, 2)
        self.conv2 = Conv2d(16, 32, 5)
        self.pool2 = MaxPool2d(2, 2)
        self.fc1 = Linear(32 * 5 * 5, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 10)

    def forward(self, x):
        x = relu(self.conv1(x))  # input(3,32,32) output(16,28,28)
        x = self.pool1(x)  # output(16，14，14)
        x = relu(self.conv2(x))  # output(32,10.10)
        x = self.pool2(x)  # output(32,5,5)
        x = x.view(-1, 32 * 5 * 5)  # output(5*5*32)
        x = relu(self.fc1(x))  # output(120)
        x = relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        return x
