import torch
from torch import nn
import math
from swin_transformer import *
from collections import OrderedDict
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        kernel_sizes = [(1, 1, 1), (3, 3, 3), (3, 3, 3), (1, 1, 1)]
        dilations = [(1, 1, 1), (1, 3, 3), (1, 6, 6), (1, 1, 1)]
        paddings = [(0, 0, 0), (1, 3, 3), (1, 6, 6), (0, 0, 0)]
        self.aspp = nn.ModuleList()
        for aspp_idx in range(len(kernel_sizes)):
            conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_sizes[aspp_idx],
                stride=1,
                dilation=dilations[aspp_idx],
                padding=paddings[aspp_idx],
                bias=True)
            self.aspp.append(conv)
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.aspp_num = len(kernel_sizes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0)
        
    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(self.aspp_num):
            inp = avg_x if (aspp_idx == self.aspp_num - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        return out
        
class Fusion(nn.Module):
    def __init__(self, input_channels):
        super(Fusion, self).__init__()
        self.weight = nn.Conv3d(input_channels, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True)

    def forward(self, x, y):
        add_weight = torch.sigmoid(self.weight(x))
        out = add_weight * x + (1 - add_weight) * y
        return out


class Saliencymodel(nn.Module):
    def __init__(self, pretrain=None):
        super(Saliencymodel, self).__init__()

        # Backbone Swin Transformer
        self.backbone = SwinTransformer3D(pretrained=pretrain)
        
        self.conv1 = nn.Conv3d(96, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.conv2 = nn.Conv3d(192, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.conv3 = nn.Conv3d(384, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        self.conv4 = nn.Conv3d(768, 192, kernel_size=(2, 1, 1), stride=(2, 1, 1))
        
        self.upsampling2 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.upsampling4 = nn.Upsample(scale_factor=(1, 4, 4), mode='trilinear')
        self.upsampling8 = nn.Upsample(scale_factor=(1, 8, 8), mode='trilinear')

        self.convs1 = nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.convs2 = nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.convs3 = nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.convs4 = nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)

        self.aspp1 = ASPP(192, 48) #out=out_channel*4
        self.aspp2 = ASPP(192, 48)
        self.aspp3 = ASPP(192, 48)
        self.aspp4 = ASPP(192, 48)

        self.fusion1 = Fusion(192)
        self.fusion2 = Fusion(192)
        self.fusion3 = Fusion(192)
        self.fusion4 = Fusion(192)

        self.convtsp1 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.Sigmoid()
        )
        self.convtsp2 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            nn.Sigmoid()
        )
        self.convtsp3 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            self.upsampling2,
            nn.Sigmoid()
        )
        self.convtsp4 = nn.Sequential(
            nn.Conv3d(192, 96, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(96, 48, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(48, 24, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1), bias=False),
            nn.ReLU(),
            self.upsampling2,
            nn.Conv3d(24, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False),
            self.upsampling4,
            nn.Sigmoid()
        )

        self.convout = nn.Sequential(
            nn.Conv3d(4, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        #Backbone
        x, [y1, y2, y3, y4] = self.backbone(x) # (96, 192, 384, 768)

        #RFP1
        r1 = self.conv1(y1)
        r2 = self.conv2(y2)
        r3 = self.conv3(y3)
        r4 = self.conv4(y4)

        t3 = self.upsampling2(r4) + r3
        c3 = self.convs3(t3)
        t2 = self.upsampling2(t3) + r2 + self.upsampling4(r4)
        c2 = self.convs2(t2)
        t1 = self.upsampling2(t2) + r1 + self.upsampling8(r4)
        c1 = self.convs1(t1)

        #RFP2
        a1 = self.aspp1(c1)
        a2 = self.aspp2(c2)
        a3 = self.aspp3(c3)
        a4 = self.aspp4(r4)

        p1 = r1 + a1
        p2 = r2 + a2
        p3 = r3 + a3
        p4 = r4 + a4

        p1 = self.convs1(p1)
        p2 = self.convs2(p2)
        p3 = self.convs3(p3)
        p4 = self.convs4(p4)

        t3 = self.upsampling2(p4) + p3
        p3 = self.convs3(t3)
        t2 = self.upsampling2(t3) + p2 + self.upsampling4(p4)
        p2 = self.convs2(t2)
        t1 = self.upsampling2(t2) + p1 + self.upsampling8(p4)
        p1 = self.convs1(t1)

        f1 = self.fusion1(p1, c1)
        f2 = self.fusion2(p2, c2)
        f3 = self.fusion3(p3, c3)
        f4 = self.fusion4(p4, r4)
        
        #RFP3
        a1 = self.aspp1(f1)
        a2 = self.aspp2(f2)
        a3 = self.aspp3(f3)
        a4 = self.aspp4(f4)

        p1 = r1 + a1
        p2 = r2 + a2
        p3 = r3 + a3
        p4 = r4 + a4

        p1 = self.convs1(p1)
        p2 = self.convs2(p2)
        p3 = self.convs3(p3)
        p4 = self.convs4(p4)

        t3 = self.upsampling2(p4) + p3
        p3 = self.convs3(t3)
        t2 = self.upsampling2(t3) + p2 + self.upsampling4(p4)
        p2 = self.convs2(t2)
        t1 = self.upsampling2(t2) + p1 + self.upsampling8(p4)
        p1 = self.convs1(t1)

        f1 = self.fusion1(p1, f1)
        f2 = self.fusion2(p2, f2)
        f3 = self.fusion3(p3, f3)
        f4 = self.fusion4(p4, f4)

        #Head
        z1 = self.convtsp1(f1)

        z2 = self.convtsp2(f2)

        z3 = self.convtsp3(f3)

        z4 = self.convtsp4(f4)
        
        z0 = self.convout(torch.cat((z1, z2, z3, z4), 1))
        z0 = z0.view(z0.size(0), z0.size(3), z0.size(4))
        return z0