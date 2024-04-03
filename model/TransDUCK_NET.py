import torch
import torch.nn as nn
from model.conv_block import ConvBlock2D

class DUCKNet(nn.Module):
    def __init__(self, input_channels, out_classes, starting_filters, kernel_initializer='he_uniform', interpolation='nearest'):
        super(DUCKNet, self).__init__()
        self.starting_filters = starting_filters

        # Encoder
        self.conv1 = nn.Conv2d(input_channels, starting_filters * 2, kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(starting_filters * 2, starting_filters * 4, kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(starting_filters * 4, starting_filters * 8, kernel_size=2, stride=2, padding=1)
        self.conv4 = nn.Conv2d(starting_filters * 8, starting_filters * 16, kernel_size=2, stride=2, padding=1)
        self.conv5 = nn.Conv2d(starting_filters * 16, starting_filters * 32, kernel_size=2, stride=2, padding=1)

        self.conv_block1 = ConvBlock2D(input_channels, starting_filters, 'duckv2', repeat=1)
        self.conv_block2 = ConvBlock2D(starting_filters * 2, starting_filters * 2, 'duckv2', repeat=1)
        self.conv_block3 = ConvBlock2D(starting_filters * 4, starting_filters * 4, 'duckv2', repeat=1)
        self.conv_block4 = ConvBlock2D(starting_filters * 8, starting_filters * 8, 'duckv2', repeat=1)
        self.conv_block5 = ConvBlock2D(starting_filters * 16, starting_filters * 16, 'duckv2', repeat=1)
        self.conv_block6 = ConvBlock2D(starting_filters * 32, starting_filters * 32, 'resnet', repeat=2)
        self.conv_block7 = ConvBlock2D(starting_filters * 32, starting_filters * 16, 'resnet', repeat=2)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(starting_filters * 16, starting_filters * 8, kernel_size=2, stride=2, padding=0)
        self.conv_block8 = ConvBlock2D(starting_filters * 16, starting_filters * 8, 'duckv2', repeat=1)

        self.upconv3 = nn.ConvTranspose2d(starting_filters * 8, starting_filters * 4, kernel_size=2, stride=2, padding=0)
        self.conv_block9 = ConvBlock2D(starting_filters * 8, starting_filters * 4, 'duckv2', repeat=1)

        self.upconv2 = nn.ConvTranspose2d(starting_filters * 4, starting_filters * 2, kernel_size=2, stride=2, padding=0)
        self.conv_block10 = ConvBlock2D(starting_filters * 4, starting_filters * 2, 'duckv2', repeat=1)

        self.upconv1 = nn.ConvTranspose2d(starting_filters * 2, starting_filters, kernel_size=2, stride=2, padding=0)
        self.conv_block11 = ConvBlock2D(starting_filters * 2, starting_filters, 'duckv2', repeat=1)

        self.upconv0 = nn.ConvTranspose2d(starting_filters, starting_filters, kernel_size=2, stride=2, padding=0)
        self.conv_block12 = ConvBlock2D(starting_filters * 2, starting_filters, 'duckv2', repeat=1)

        self.final_conv = nn.Conv2d(starting_filters, out_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        p1 = self.conv1(x)
        p2 = self.conv2(p1)
        p3 = self.conv3(p2)
        p4 = self.conv4(p3)
        p5 = self.conv5(p4)

        t0 = self.conv_block1(x)

        l1i = self.conv1(t0)
        s1 = p1 + l1i
        t1 = self.conv_block2(s1)

        l2i = self.conv2(t1)
        s2 = p2 + l2i
        t2 = self.conv_block3(s2)

        l3i = self.conv3(t2)
        s3 = p3 + l3i
        t3 = self.conv_block4(s3)

        l4i = self.conv4(t3)
        s4 = p4 + l4i
        t4 = self.conv_block5(s4)

        l5i = self.conv5(t4)
        s5 = p5 + l5i
        t51 = self.conv_block6(s5)
        t53 = self.conv_block7(t51)

        # Decoder
        l5o = self.upconv4(t53)
        c4 = l5o + t4
        q4 = self.conv_block8(c4)

        l4o = self.upconv3(q4)
        c3 = l4o + t3
        q3 = self.conv_block9(c3)

        l3o = self.upconv2(q3)
        c2 = l3o + t2
        q6 = self.conv_block10(c2)

        l2o = self.upconv1(q6)
        c1 = l2o + t1
        q1 = self.conv_block11(c1)

        l1o = self.upconv0(q1)
        c0 = l1o + t0
        z1 = self.conv_block12(c0)

        output = self.final_conv(z1)

        return output
    
