import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SKConv(nn.Module):
    def __init__(self, channels, branches=2, reduce=2, stride=1, len=32):
        super(SKConv, self).__init__()
        len = max(int(channels // reduce), len)
        self.convs = nn.ModuleList([])
        for i in range(branches):
            self.convs.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i,
                           bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(channels, len, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(len),
            nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([])
        for i in range(branches):
            self.fcs.append(
                nn.Conv2d(len, channels, kernel_size=1, stride=1,bias=False)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = [conv(x) for conv in self.convs]
        x = torch.stack(x, dim=1)
        attention = torch.sum(x, dim=1)
        attention = self.gap(attention)
        attention = self.fc(attention)
        attention = [fc(attention) for fc in self.fcs]
        attention = torch.stack(attention, dim=1)
        attention = self.softmax(attention)
        x = torch.sum(x * attention, dim=1)
        return x

class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k, s): # 如果s==1 则c1=c2 才能shortcut
        super(GhostBottleneck, self).__init__()
        c_ = c2 // 2
        # print(c1, c2, c_, k, s)
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        # print(self.conv(x))
        # print(self.shortcut(x))
        return self.conv(x) + self.shortcut(x)

class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)
        # self.cv1 = Conv(c1, c_, k, s, g, act)
        # self.cv2 = Conv(c_, c_, 5, 1, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)

def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


# ----------------------------
# Audio Classification Model
# ----------------------------
class AudioClassifier(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()

        # self.sepConv = depthwise_separable_conv(32, 32)

        # First Convolution Block with Relu and Batch Norm. 标准卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            # nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2), stride=(2, 2))
        )
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2))
        # self.relu1 = nn.ReLU()
        # self.bn1 = nn.BatchNorm2d(32)
        # self.mp1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        # conv_layers += [self.conv1, self.bn1, self.relu1, self.mp1]
        # self.block1 = nn.Sequential(self.conv1, self.bn1, self.relu1, self.mp1)

        # Second Convolution Block. Ghost Bottleneck
        # self.conv2 = GhostBottleneck(32, 32, 3, 1)
        self.conv2 = SKConv(32)
        # self.conv2 = SELayer(32)

        # Third Convolution Block. Ghost Bottleneck
        self.conv3 = GhostBottleneck(32, 32, 3, 2)

        # Fourth Convolution Block. Ghost Bottleneck
        # self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
        self.conv4 = GhostBottleneck(32, 32, 3, 2)
        # self.conv4 = SKConv(32, branches=2, reduce=2, stride=1, len=32)

        # self.conv5 = SKConv(32)
        # Linear Classifier
        # self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        

        self.dense1 = nn.Linear(in_features=32, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=2)
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.dp = nn.Dropout(p=0.5)
        # self.conv5 = GhostBottleneck(32, 32, 3, 1)

    


    def forward(self, x):

        # x = self.block1(x)
        x = self.conv1(x)
        # print('1',x.shape)
        x = self.conv2(x)
        # print('2',x.shape)
        x = self.conv3(x)
        # x = self.SK(x)
        # x = self.dp(x)
        # print('3.0',x.shape)
        x = self.conv4(x)
        # print('3.1',x.shape)
        # x = self.conv2(x)
        # x = self.conv5(x)
        x = self.ap(x)
        x_all = x.view(x.shape[0], -1)
        # print('4',x_all.shape)
        # x_all = self.dp(x_all)
        x_all = self.dense1(x_all)
        # print('5',x_all.shape)
        x_all = self.dp(x_all)
        x_all = self.dense2(x_all)
        # print('6',x_all.shape)

        # Final output
        return x_all
