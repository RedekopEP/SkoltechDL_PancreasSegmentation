
import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up_spatAtten(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up_spatAtten, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

        self.W_g = nn.Sequential(
            nn.Conv2d(in_ch // 2, out_ch // 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch // 2)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(in_ch // 2, out_ch // 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch // 2)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(out_ch // 2, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    class up_spatAtten(nn.Module):
        def __init__(self, in_ch, out_ch, bilinear=True):
            super(up_spatAtten, self).__init__()

            #  would be a nice idea if the upsampling could be learned too,
            #  but my machine do not have enough memory to handle all those weights
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

            self.conv = double_conv(in_ch, out_ch)

            self.W_g = nn.Sequential(
                nn.Conv2d(in_ch // 2, out_ch // 2, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(out_ch // 2)
            )

            self.W_x = nn.Sequential(
                nn.Conv2d(in_ch // 2, out_ch // 2, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(out_ch // 2)
            )

            self.psi = nn.Sequential(
                nn.Conv2d(out_ch // 2, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

            self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):

        x1 = self.up(x1)

        g = self.W_g(x2)
        x = self.W_x(x1)
        psi = self.relu(g + x)
        psi = self.psi(psi)

        g2 = x1 * psi

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([g2, x1], dim=1)
        x = self.conv(x)
        return x


class up_chanAtten(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=16, bilinear=True):
        super(up_chanAtten, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.cSE = nn.Sequential(
            nn.Linear(in_ch // 2, in_ch // 2 * reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // 2 * reduction, in_ch // 2),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):

        x1 = self.up(x1)

        x11 = self.avg_pool(x1).view(x1.shape[0], x1.shape[1])
        x11 = self.cSE(x11).view(x1.shape[0], x1.shape[1], 1, 1)

        g1 = self.avg_pool(x2).view(x2.shape[0], x2.shape[1])
        g1 = self.cSE(g1).view(x2.shape[0], x2.shape[1], 1, 1)

        psi = self.relu(g1 + x11)

        psi1 = self.avg_pool(psi).view(psi.shape[0], psi.shape[1])
        psi2 = self.cSE(psi1).view(psi.shape[0], psi.shape[1], 1, 1)

        g2 = x1 * psi2

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([g2, x1], dim=1)
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)

        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class SpatialAttenUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(SpatialAttenUNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up_spatAtten(1024, 256)
        self.up2 = up_spatAtten(512, 128)
        self.up3 = up_spatAtten(256, 64)
        self.up4 = up_spatAtten(128, 64)

        self.outc = outconv(64, n_classes)

    def forward(self, x):
        print(x.shape)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class ChanAttenUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(ChanAttenUNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up_chanAtten(1024, 256)
        self.up2 = up_chanAtten(512, 128)
        self.up3 = up_chanAtten(256, 64)
        self.up4 = up_chanAtten(128, 64)

        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

