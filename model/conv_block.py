import torch
import torch.nn as nn

class ConvBlock2D(nn.Module):
    def __init__(self, block_type, in_channels, out_channels, kernel_size=3, repeat=1, dilation_rate=1):
        super(ConvBlock2D, self).__init__()
        self.block_type = block_type
        self.repeat = repeat
        if block_type == 'separated':
            self.conv_blocks = nn.ModuleList([SeparatedConv2D(in_channels, out_channels, kernel_size, padding='same') for _ in range(repeat)])
        elif block_type == 'duckv2':
            self.conv_blocks = nn.ModuleList([DUCKv2Conv2D(in_channels, out_channels, kernel_size) for _ in range(repeat)])
        elif block_type == 'midscope':
            self.conv_blocks = nn.ModuleList([MidScopeConv2D(in_channels, out_channels, kernel_size) for _ in range(repeat)])
        elif block_type == 'widescope':
            self.conv_blocks = nn.ModuleList([WideScopeConv2D(in_channels, out_channels, kernel_size) for _ in range(repeat)])
        elif block_type == 'resnet':
            self.conv_blocks = nn.ModuleList([ResNetConv2D(in_channels, out_channels, kernel_size, dilation_rate) for _ in range(repeat)])
        elif block_type == 'conv':
            self.conv_blocks = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size, padding='same') for _ in range(repeat)])
        elif block_type == 'double_convolution':
            self.conv_blocks = nn.ModuleList([DoubleConvolutionWithBatchNorm(in_channels, out_channels, kernel_size, dilation_rate) for _ in range(repeat)])

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        return x

class SeparatedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same'):
        super(SeparatedConv2D, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1), padding=padding)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), padding=padding)

    def forward(self, x):
        x = self.depthwise(x)
        x = nn.BatchNorm2d(x.size(1))(x)
        x = self.pointwise(x)
        x = nn.BatchNorm2d(x.size(1))(x)
        return x

class DUCKv2Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DUCKv2Conv2D, self).__init__()
        self.conv1 = WideScopeConv2D(in_channels, out_channels, kernel_size)
        self.conv2 = MidScopeConv2D(in_channels, out_channels, kernel_size)
        self.conv3 = ConvBlock2D('resnet', in_channels, out_channels, kernel_size, repeat=1)
        self.conv4 = ConvBlock2D('resnet', in_channels, out_channels, kernel_size, repeat=2)
        self.conv5 = ConvBlock2D('resnet', in_channels, out_channels, kernel_size, repeat=3)
        self.conv6 = SeparatedConv2D(in_channels, out_channels, kernel_size=6, padding='same')

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)
        x6 = self.conv6(x)
        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        x = nn.BatchNorm2d(x.size(1))(x)
        return x

class MidScopeConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(MidScopeConv2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same', dilation=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same', dilation=2)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.BatchNorm2d(x.size(1))(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.BatchNorm2d(x.size(1))(x)
        x = nn.ReLU()(x)
        return x

class WideScopeConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(WideScopeConv2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same', dilation=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same', dilation=2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same', dilation=3)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.BatchNorm2d(x.size(1))(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.BatchNorm2d(x.size(1))(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.BatchNorm2d(x.size(1))(x)
        x = nn.ReLU()(x)
        return x

class ResNetConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate):
        super(ResNetConv2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same')
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same', dilation=dilation_rate)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same', dilation=dilation_rate)

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.conv2(x)
        x = nn.BatchNorm2d(x.size(1))(x)
        x = nn.ReLU()(x)
        x = self.conv3(x)
        x = nn.BatchNorm2d(x.size(1))(x)
        x = x + x1
        x = nn.ReLU()(x)
        return x

class DoubleConvolutionWithBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate):
        super(DoubleConvolutionWithBatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same', dilation=dilation_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding='same', dilation=dilation_rate)
        
    def forward(self, x):
        x = self.conv1(x)
        x = nn.BatchNorm2d(x.size(1))(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.BatchNorm2d(x.size(1))(x)
        x = nn.ReLU()(x)
        return x
    
    