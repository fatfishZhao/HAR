import torch
import torch.nn as nn
import torch.autograd as autograd

class BasicConv1d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(out_channel,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FCN(nn.Module):
    def __init__(self, num_classes=5):
        super(FCN, self).__init__()
        self.num_classes = num_classes

        self.conv0 = BasicConv1d(in_channel=6, out_channel=128, kernel_size=3, stride=1, padding=1)
        self.conv1 = BasicConv1d(in_channel=128, out_channel=256, kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConv1d(in_channel=256, out_channel=128, kernel_size=3, stride=1, padding=1)
        # self.gp = nn.AdaptiveMaxPool1d(1)
        self.gp = nn.MaxPool1d(256) # need to be changed when the length of window size changed
        self.fc = nn.Linear(128, self.num_classes)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        feature = self.gp(x)
        feature = feature.view(feature.shape[0], -1)
        y = self.fc(feature)

        return y

if __name__ == '__main__':
    input = autograd.Variable(torch.randn(20, 256, 6))
    input = input.permute(0, 2, 1)
    m = FCN(6)
    y = m(input)
    print(y)
