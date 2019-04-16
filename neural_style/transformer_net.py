import torch


class TransformerNet(torch.nn.Module):
    def __init__(self, alpha = 1.0, use_small_network=False, use_separable_conv=False, image_size=256):
        super(TransformerNet, self).__init__()
        self.use_small_network = use_small_network
        #global image_size
        if use_small_network:
            # Initial convolution layers
            self.conv1 = ConvLayer(3, int(alpha * 32), kernel_size=9, stride=1)
            self.in1 = torch.nn.InstanceNorm2d(int(alpha * 32), affine=True)
            self.conv2 = ConvLayer(int(alpha * 32), int(alpha * 32), kernel_size=3, stride=2, use_separable_conv=use_separable_conv)
            self.in2 = torch.nn.InstanceNorm2d(int(alpha * 32), affine=True)
            self.conv3 = ConvLayer(int(alpha * 32), int(alpha * 32), kernel_size=3, stride=2, use_separable_conv=use_separable_conv)
            self.in3 = torch.nn.InstanceNorm2d(int(alpha * 32), affine=True)
            # Residual layers
            self.res1 = ResidualBlock(int(alpha * 32))
            self.res2 = ResidualBlock(int(alpha * 32))
            self.res3 = ResidualBlock(int(alpha * 32))
            # Upsampling Layers
            self.deconv1 = UpsampleConvLayer(int(alpha * 32), int(alpha * 32), kernel_size=3, stride=1, upsample=2, use_separable_conv=use_separable_conv)
            self.in4 = torch.nn.InstanceNorm2d(int(alpha * 32), affine=True)
            self.deconv2 = UpsampleConvLayer(int(alpha * 32), int(alpha * 32), kernel_size=3, stride=1, upsample=2, use_separable_conv=use_separable_conv)
            self.in5 = torch.nn.InstanceNorm2d(int(alpha * 32), affine=True)
            self.deconv3 = ConvLayer(int(alpha * 32), 3, kernel_size=9, stride=1)
            # Non-linearities
            self.relu = torch.nn.ReLU()
        else:
            # Initial convolution layers
            self.conv1 = ConvLayer(3, int(alpha * 32), kernel_size=9, stride=1)
            self.in1 = torch.nn.InstanceNorm2d(int(alpha * 32), affine=True)
            self.conv2 = ConvLayer(int(alpha * 32), int(alpha * 64), kernel_size=3, stride=2, use_separable_conv=use_separable_conv)
            self.in2 = torch.nn.InstanceNorm2d(int(alpha * 64), affine=True)
            self.conv3 = ConvLayer(int(alpha * 64), int(alpha * 128), kernel_size=3, stride=2, use_separable_conv=use_separable_conv)
            self.in3 = torch.nn.InstanceNorm2d(int(alpha * 128), affine=True)
            # Residual layers
            self.res1 = ResidualBlock(int(alpha * 128))
            self.res2 = ResidualBlock(int(alpha * 128))
            self.res3 = ResidualBlock(int(alpha * 128))
            self.res4 = ResidualBlock(int(alpha * 128))
            self.res5 = ResidualBlock(int(alpha * 128))
            # Upsampling Layers
            self.deconv1 = UpsampleConvLayer(int(alpha * 128), int(alpha * 64), kernel_size=3, stride=1, upsample=2, use_separable_conv=use_separable_conv)
            self.in4 = torch.nn.InstanceNorm2d(int(alpha * 64), affine=True)
            self.deconv2 = UpsampleConvLayer(int(alpha * 64), int(alpha * 32), kernel_size=3, stride=1, upsample=2, use_separable_conv=use_separable_conv)
            self.in5 = torch.nn.InstanceNorm2d(int(alpha * 32), affine=True)
            self.deconv3 = ConvLayer(int(alpha * 32), 3, kernel_size=9, stride=1)
            # Non-linearities
            self.relu = torch.nn.ReLU()

    def forward(self, X):
        if self.use_small_network:
            y = self.relu(self.in1(self.conv1(X)))
            y = self.relu(self.in2(self.conv2(y)))
            y = self.relu(self.in3(self.conv3(y)))
            y = self.res1(y)
            y = self.res2(y)
            y = self.res3(y)
            y = self.relu(self.in4(self.deconv1(y)))
            y = self.relu(self.in5(self.deconv2(y)))
            y = self.deconv3(y)
        else:
            y = self.relu(self.in1(self.conv1(X)))
            y = self.relu(self.in2(self.conv2(y)))
            y = self.relu(self.in3(self.conv3(y)))
            y = self.res1(y)
            y = self.res2(y)
            y = self.res3(y)
            y = self.res4(y)
            y = self.res5(y)
            y = self.relu(self.in4(self.deconv1(y)))
            y = self.relu(self.in5(self.deconv2(y)))
            y = self.deconv3(y)
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_separable_conv=False):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        # padding = 0
        # reflection_padding = kernel_size // 2
        # self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        if use_separable_conv:
            self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=in_channels, padding=padding)
        else:
            self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)

    def forward(self, x):
        out = x.clone()
        # out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, use_separable_conv=True):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        padding = kernel_size // 2
        # padding = 0
        # reflection_padding = kernel_size // 2
        # self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        if use_separable_conv:
            self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=in_channels, padding=padding)
        else:
            self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        # out = self.reflection_pad(x_in)
        out = x_in
        out = self.conv2d(out)
        return out
