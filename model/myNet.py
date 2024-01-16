import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import glob
import imageio
import natsort
from pytorch_msssim import ssim
import os
import os.path
import multiprocessing
import scipy.io as scio
from PIL import Image
import cv2
import matplotlib
import scipy.misc

device = 'cpu'



class _LFConv1(nn.Module):
    def __init__(self):
        super(_LFConv1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _LFConv2(nn.Module):
    def __init__(self):
        super(_LFConv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(2),
            nn.ReLU(True),
            nn.Conv2d(2, 4, 7, 1, 3),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.Conv2d(4, 8, 7, 1, 3),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 7, 1, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 7, 1, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _HFConv1(nn.Module):
    def __init__(self):
        super(_HFConv1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _HFConv2(nn.Module):
    def __init__(self):
        super(_HFConv2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.pool = nn.MaxPool2d(2, 2)
        # self.up = nn.Upsample(scale_factor=2)

    def upsample(self, x, size):
        return nn.functional.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.upsample(x4, (256, 256))
        x1_2 = self.conv2(x1)
        x5_2 = self.conv6(x5)
        cat = torch.cat((x1_2, x5_2), 1)
        x6 = self.conv4(cat)

        return x6


def denorm(mean=[0, 0, 0], std=[1, 1, 1], tensor=None):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


class forOriginal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(forOriginal, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class DsConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(DsConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class downsample(nn.Module):
    def __init__(self, dw_channels1=16, dw_channels2=32, out_channels=64, **kwargs):
        super(downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, dw_channels1, 3, 2),
            nn.BatchNorm2d(dw_channels1),
            nn.ReLU(True)
        )
        self.dsconv1 = DsConv(dw_channels1, dw_channels2, 2)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.dsconv1(x1)

        return x1, x2


class SegConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(SegConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class SegConv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, **kwargs):
        super(SegConv2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class PoolandUp(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(PoolandUp, self).__init__()
        inter_channels = int(in_channels / 2)
        self.conv1 = SegConv(in_channels, inter_channels, 1, **kwargs)
        self.out = SegConv(in_channels, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return nn.functional.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        p1 = self.pool(x, int(size[0] / 2))
        cp1 = self.conv1(p1)
        ucp1 = self.upsample(cp1, size)

        x = self.upsample(self.conv1(x), size)

        x = torch.cat([x, ucp1], dim=1)
        x = self.out(x)

        return x


class SeFeature(nn.Module):
    def __init__(self, in_channels, out_channels, t=2, stride=2, **kwargs):
        super(SeFeature, self).__init__()
        self.block = nn.Sequential(
            SegConv(in_channels, in_channels * t, 1),
            SegConv2(in_channels * t, in_channels * t),
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.pool_up = PoolandUp(32, 32)

    def forward(self, x):
        out1 = self.block(x)
        out = self.pool_up(out1)
        return out


class segment(nn.Module):
    def __init__(self, in_channels, num_classes, stride=1, **kwargs):
        super(segment, self).__init__()
        self.conv1 = DsConv(in_channels, in_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels, num_classes, 1)
        )

    def forward(self, x):
        size = x.size()[2:]
        x1 = self.conv1(x)
        x2 = self.conv(x1)

        return x2


class AffineTransform(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x):
        return self.alpha * x + self.beta


def net():
    # define the network
    class DeepTreeFuse(nn.Module):
        def __init__(self, num_classes=3):
            super(DeepTreeFuse, self).__init__()

            #####one lf layer 1#####
            self.one_lf1 = _LFConv1()
            self.one_lf2 = _LFConv2()
            #####one hf layers#####
            self.one_hf1 = _HFConv1()
            self.one_hf2 = _HFConv2()

            #####two lf layer 1#####
            self.two_lf1 = _LFConv1()
            self.two_lf2 = _LFConv2()
            #####two hf layers#####
            self.two_hf1 = _HFConv1()
            self.two_hf2 = _HFConv2()

            self.l_o = forOriginal(1, 16, 7, 1, 3)
            self.h_o = forOriginal(1, 16, 3, 1, 3)

            # lf reconstruction
            self.lf_recons = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1,
                          padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1,
                          padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(True)
            )  # output shape (,16,256,256)

            # hf reconstruction
            self.hf_recons = nn.Sequential(
                nn.Conv2d(in_channels=256, out_channels=128, kernel_size=5, stride=1,
                          padding=2),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1,
                          padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1,
                          padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1,
                          padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
            )  # output shape (,16,256,256)

            # final reconstruction
            self.recon = nn.Sequential(  # input shape (,16, 256, 256)
                nn.Conv2d(in_channels=16, out_channels=3, kernel_size=5, stride=1,
                          padding=2))  # output shape (,3,256,256)
            self.hf2con = nn.Sequential(  # input shape (,16, 256, 256)
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1,
                          padding=2))
            self.reconse = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1,
                          padding=2))
            self.re = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1,
                          padding=1))

            # semantic feature extract
            self.downsample = downsample()
            self.globalfeature = SeFeature(32, 32)
            self.otherconv = SegConv2(32, 32)
            self.merge = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,
                          padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=False)
            )

            self.segclass = segment(32, num_classes=3)
            self.dropandplt = nn.Sequential(
                nn.Conv2d(32, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(16, num_classes, 1)
            )
            self.x0 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(True))
            self.x1 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(True))
            self.x2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(True))
            self.xxx = nn.Sequential(nn.Conv2d(96, 64, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(True),
                                     nn.Conv2d(64, 32, 3, padding=1, bias=False),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(True)
                                     )
            self.aff = AffineTransform(256)

        def upsample(self, x, size):
            return nn.functional.interpolate(x, size, mode='bilinear', align_corners=True)

        def tensor_max(self, tensors):
            max_tensor = None
            for i, tensor in enumerate(tensors):
                if i == 0:
                    max_tensor = tensor
                else:
                    max_tensor = torch.max(max_tensor, tensor)
            return max_tensor

        def obtain_to_LHF(self, tensor):
            # lf
            x = tensor
            lf1 = self.one_lf1(x)
            lf2 = self.one_lf2(x)
            x1 = torch.cat((lf1, lf2), 1)
            lf_x = self.lf_recons(x1)
            # hf
            hf1 = self.one_hf1(x)
            hf2 = self.one_hf2(x)
            x2 = torch.cat((hf1, hf2), 1)
            hf_x = self.hf_recons(x2)
            x_change_c_ol = self.l_o(x)
            x_change_c_oh = self.h_o(x)
            outx = lf_x + hf_x

            return lf_x, hf_x, outx, x_change_c_ol, x_change_c_oh

        def semantic_one(self, tensor):
            x = tensor
            sx, se = self.downsample(x)
            s1 = self.globalfeature(se)
            s2 = self.otherconv(se)
            s = s1 + s2
            out_s = self.merge(s)
            out_up = self.upsample(out_s, (128, 128))
            sclass = self.segclass(out_up)
            size = x.size()[2:]
            sclass = self.upsample(sclass, size)
            result_S = sclass
            x0 = self.x0(x)
            sx = self.x1(sx)
            se2 = self.x2(se)
            sx = self.upsample(sx, size)
            se2 = self.upsample(se2, size)
            x0 = self.dropandplt(x0)  # self.segclass(x0)
            sx = self.segclass(sx)
            se2 = self.segclass(se2)
            x3_out = x0 + sx + se2
            total_seg1 = result_S + x3_out

            return total_seg1

        def doFuse(self, t1, t2, t3, t4, xl, xh, yl, yh, cx_l, cx_h, cy_l, cy_h):
            lhf_x = t1
            lhf_y = t2
            se_x = t3
            se_y = t4
            xh = self.aff(xh)
            xl = self.aff(xl)
            yh = self.aff(yh)
            yl = self.aff(yl)

            HF = xh + yh
            LF = xl + yl
            fu_lf_hf = self.tensor_max([HF, LF])

            lf_hf = self.recon(fu_lf_hf)
            lf_hf = self.aff(lf_hf)

            out_se = (se_x + se_y) / 2
            out_se = self.aff(out_se)

            fuseout = (lf_hf + out_se) / 2
            fuseout = torch.tanh(fuseout)

            return fuseout

        def forward(self, x, y):  # lf,hf,pyramid and reconstruction of mul-type
            # print(torch.equal(x, y))
            ## extract deep features
            # __________________ one type __________________

            lf_x, hf_x, lhf_x, cx_l, cx_h = self.obtain_to_LHF(x)
            se_x = self.semantic_one(x)  # semantic
            # print('one type result out **********')

            # __________________ two type __________________
            lf_y, hf_y, lhf_y, cy_l, cy_h = self.obtain_to_LHF(y)
            se_y = self.semantic_one(y)
            # print('the other type result out **********')

            ## to do fusing
            fuseout = self.doFuse(lhf_x, lhf_y, se_x, se_y, lf_x, hf_x, lf_y, hf_y, cx_l, cx_h, cy_l, cy_h)
            fuseout = self.re(fuseout)

            return fuseout

    dtn = DeepTreeFuse().to(device)
    dtn = dtn.float()
    # print(dtn)

    return dtn






if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #print(device)
    

