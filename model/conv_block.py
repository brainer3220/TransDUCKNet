import logging
import torch
import torch.nn as nn

class ConvBlock2D(nn.Module):
    def __init__(self, block_type, in_channels, out_channels, kernel_size=3, repeat=1, dilation_rate=1):
        super(ConvBlock2D, self).__init__()
        self.block_type = block_type
        self.repeat = repeat
        modules = []
        for _ in range(repeat):
            if block_type == 'separated':
                modules.append(SeparatedConv2D(in_channels, out_channels, kernel_size, padding='same'))
            elif block_type == 'duckv2':
                modules.append(DUCKv2Conv2D(in_channels, out_channels, kernel_size))
            elif block_type == 'midscope':
                modules.append(MidScopeConv2D(in_channels, out_channels, kernel_size))
            elif block_type == 'widescope':
                modules.append(WideScopeConv2D(in_channels, out_channels, kernel_size))
            elif block_type == 'resnet':
                modules.append(ResNetConv2D(in_channels, out_channels, dilation_rate))
            elif block_type == 'conv':
                modules.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'))
            elif block_type == 'double_convolution':
                modules.append(DoubleConvolutionWithBatchNorm(in_channels, out_channels, kernel_size, dilation_rate))
            in_channels = out_channels  # 업데이트된 out_channels를 다음 블록의 in_channels로 설정
        self.conv_blocks = nn.ModuleList(modules)

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
        self.conv1 = ConvBlock2D('widescope', in_channels, out_channels, kernel_size, repeat=1)
        self.conv2 = ConvBlock2D('midscope', in_channels, out_channels, kernel_size, repeat=1)
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
        
        x = x1 + x2 + x3 + x4 + x5 + x6
        logging.debug(f"DUCKv2Conv2D: x1: {x1.size()}, x2: {x2.size()}, x3: {x3.size()}, x4: {x4.size()}, x5: {x5.size()}, x6: {x6.size()}, x: {x.size()}")
        x = nn.BatchNorm2d(x.size(1))(x)
        x = nn.ReLU()(x)
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
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super(ResNetConv2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding="same", dilation=dilation_rate, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", dilation=dilation_rate, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", dilation=dilation_rate, bias=False)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(x1)

        x = self.conv2(x)
        
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn2(x)

        x_final = x + x1
        x_final = self.bn3(x_final)
        x_final = self.relu(x_final)

        return x_final

    
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
        
        