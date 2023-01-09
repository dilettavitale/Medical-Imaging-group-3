import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from segmentation_models_pytorch.encoders import get_encoder

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
    
    def forward(self, x):
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        return out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            #DoubleConv(in_channels, out_channels)
            DenseBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        x = self.conv(x)
        return self.up(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DenseBlock(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(3072, 1792, bilinear)
        self.up2 = Up(2304, 512, bilinear)
        self.up3 = Up(768, 256, bilinear)
        self.up4 = Up(320, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.encoder= get_encoder(
            name = "se_resnet50",
            in_channels=3,
            depth= 5,
            weights= None,
        )

        self.ann1_1 = nn.Linear(2048 , 512)
        self.ann1_2 = nn.Linear(512, 7)
        self.ann2_1 = nn.Linear(2048 , 512)
        self.ann2_2 = nn.Linear(512, 1)
        self.maxpool2d = nn.MaxPool2d(8)
        
        self.weights = torch.nn.Parameter(torch.ones(3).float())

    def forward(self, x):

        features = self.encoder(x)
        x0, x1, x2, x3, x4, x5 = features
   
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        a=self.maxpool2d(x5)
        a= a.view(a.size(0), -1)
        
        #a = x5.view(x5.size(0), -1)
        a1 = self.ann1_1(a)
        a1 = F.relu(a1)
        a1 = self.ann1_2(a1)
        
        lab = a1
        lab = torch.exp(lab)
        a2 = self.ann2_1(a)
        a2 = F.relu(a2)        
        a2 = self.ann2_2(a2)
        intensity = a2
        return [output, lab, intensity]

    def get_last_shared_layer(self):
        stages = self.encoder.get_stages()
        return stages[5]
