"""Attention U-Net (Oktay et al. 2018)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """g = gating signal (decoder), x = skip connection (encoder)."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv2d(F_g, F_int, 1, bias=False),
                                  nn.BatchNorm2d(F_int))
        self.W_x  = nn.Sequential(nn.Conv2d(F_l, F_int, 1, bias=False),
                                  nn.BatchNorm2d(F_int))
        self.psi  = nn.Sequential(nn.Conv2d(F_int, 1, 1, bias=False),
                                  nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1, x1 = self.W_g(g), self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:],
                               mode='bilinear', align_corners=False)
        return x * self.psi(self.relu(g1 + x1))


class AttentionUNet(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1):
        super().__init__()
        def cb(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True))
        def ub(ic, oc):
            return nn.Sequential(nn.ConvTranspose2d(ic, oc, 2, stride=2),
                                 nn.ReLU(inplace=True))

        self.e1, self.e2 = cb(in_ch, 64), cb(64, 128)
        self.e3, self.e4 = cb(128, 256),  cb(256, 512)
        self.center      = cb(512, 1024)
        self.a4 = AttentionBlock(1024, 512, 256)
        self.a3 = AttentionBlock(512,  256, 128)
        self.a2 = AttentionBlock(256,  128,  64)
        self.a1 = AttentionBlock(128,   64,  32)
        self.d4 = ub(1024+512, 512)
        self.d3 = ub(512+256,  256)
        self.d2 = ub(256+128,  128)
        self.d1 = ub(128+64,    64)
        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        orig = x.shape[2:]
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        c  = self.center(F.max_pool2d(e4, 2))

        a4 = self.a4(c, e4)
        d4 = self.d4(torch.cat([F.interpolate(c,  e4.shape[2:], mode='bilinear', align_corners=False), a4], dim=1))
        a3 = self.a3(d4, e3)
        d3 = self.d3(torch.cat([F.interpolate(d4, e3.shape[2:], mode='bilinear', align_corners=False), a3], dim=1))
        a2 = self.a2(d3, e2)
        d2 = self.d2(torch.cat([F.interpolate(d3, e2.shape[2:], mode='bilinear', align_corners=False), a2], dim=1))
        a1 = self.a1(d2, e1)
        d1 = self.d1(torch.cat([F.interpolate(d2, e1.shape[2:], mode='bilinear', align_corners=False), a1], dim=1))

        return F.interpolate(self.final(d1), size=orig,
                             mode='bilinear', align_corners=False)
