import torch
import torch.nn as nn

class Vanila_Conv_no_pool(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = activation(c2, act_num=3)
        # self.act = self.default_act if ahaoct is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))


def Probility_refine(x1, x2):
    w = x1 * x2
    w_sum = x1 * x2 + (1. - x1) * (1. - x2)
    return w / (w_sum + 1e-6)


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class activation(nn.ReLU):
    def __init__(self, dim, act_num=3, deploy=False):
        super(activation, self).__init__()
        self.deploy = deploy
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num * 2 + 1, act_num * 2 + 1))
        self.bias = None
        self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        self.dim = dim
        self.act_num = act_num

    def forward(self, x):
        if self.deploy:
            return torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, self.bias, padding=(self.act_num * 2 + 1) // 2, groups=self.dim)
        else:
            return self.bn(torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, padding=(self.act_num * 2 + 1) // 2, groups=self.dim))

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std

    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        self.weight.data = kernel
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        self.bias.data = bias
        self.__delattr__('bn')
        self.deploy = True


class Vanila_Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = activation(c2, act_num=3)
        self.downsample = Cut(c2, c2)
        # self.act = self.default_act if ahaoct is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.downsample(self.act(self.bn(self.conv(x))))

    def forward_fuse(self, x):
        return self.downsample(self.act((self.conv(x))))





class FFE(nn.Module):
    def __init__(self, inchannel_low, inchannel_high, ratio=16):
        super(FFE, self).__init__()
        self.downsample = Cut(inchannel_low, inchannel_low)

        self.conv_block = nn.Sequential(
            Vanila_Conv_no_pool(inchannel_high, inchannel_high // ratio, 3),
            nn.Conv2d(inchannel_high // ratio, 1, 1, padding=0),
            nn.Sigmoid()
        )

        self.modify_low = Vanila_Conv_no_pool(inchannel_low, inchannel_high, 1)
        self.modify_high = Vanila_Conv_no_pool(inchannel_high, inchannel_high, 1)

        self.conv_sum = Vanila_Conv_no_pool(inchannel_high, inchannel_high, 1)

    def forward(self, x_low, x_high):
        b1, c1, w1, h1 = x_low.size()
        b2, c2, w2, h2 = x_high.size()
        if (w1, h2) != (w2, h2):
            x_low = self.downsample(x_low)
        att_low = self.conv_block(x_high)
        x_low = self.modify_low(x_low * att_low)

        att_high = self.conv_block(x_low)
        x_high = self.modify_high(x_high * att_high)

        x = self.conv_sum(x_high + x_low)
        return x


# CutD
class Cut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_fusion = nn.Conv2d(in_channels * 5, out_channels, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]  # x = [B, C, H/2, W/2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3, self.maxpool(x)], dim=1)  # x = [B, 4*C, H/2, W/2]
        x = self.act(self.batch_norm(self.conv_fusion(x)))
        return x


class FFD(nn.Module):
    def __init__(self, in_spatial_low, in_spatial_high, in_prior):
        super(FFD, self).__init__()
        print("feature fusion为FFD")
        self.conv_block_low = nn.Sequential(
            Vanila_Conv_no_pool(in_spatial_low, in_spatial_low // 16, 1),
            nn.Conv2d(in_spatial_low // 16, 1, 1, padding=0),
            nn.Sigmoid()
        )

        self.conv_block_high = nn.Sequential(
            Vanila_Conv_no_pool(in_spatial_high, in_spatial_high // 16, 1),
            nn.Conv2d(in_spatial_high // 16, 1, 1, padding=0),
            nn.Sigmoid()
        )
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(3, 1, 1, padding=0),
            nn.Sigmoid())

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_channel = nn.Sequential(
            nn.Conv2d(in_spatial_low, in_prior, 1, padding=0),
        )
        self.sig = nn.Sigmoid()

        self.Up_to_2 = nn.Upsample(scale_factor=2)

        self.conv_final = Vanila_Conv_no_pool(in_spatial_low, in_prior, 1)
        self.conv_concat = Vanila_Conv_no_pool(in_spatial_low + in_spatial_high, in_spatial_low, 1)

        self.final_attention = nn.Sequential(
            nn.Conv2d(in_prior, in_prior // 4, 3, 1, padding=1),
            nn.BatchNorm2d(in_prior // 4),
            nn.ReLU(),
            nn.Conv2d(in_prior // 4, 1, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x_spatial_low, x_spatial_high):
        b1, c1, w1, h1 = x_spatial_low.size()
        b2, c2, w2, h2 = x_spatial_high.size()
        if (w1, h2) != (w2, h2):
            x_spatial_high = self.Up_to_2(x_spatial_high)

        x_low_map = self.conv_block_low(x_spatial_low)
        x_high_map = self.conv_block_high(x_spatial_high)
        spatial_attention_map = self.conv_spatial(
            torch.cat([x_low_map, x_high_map, Probility_refine(x_low_map, x_high_map)], 1))

        x_spatial_low = self.conv_concat(torch.cat((x_spatial_low, x_spatial_high), 1)) * spatial_attention_map
        # x_spatial_low = self.Up_to_2(x_spatial_low)   这里进行了一次上采样，把45变成了90
        channel_attention = self.sig(
            self.conv_channel(self.avg_pool(x_spatial_low)) + self.conv_channel(self.max_pool(x_spatial_low)))

        out = self.conv_final(x_spatial_low) * channel_attention

        return out


class FFD_final(nn.Module):
    def __init__(self, in_spatial_low, in_spatial_high, in_prior):
        super(FFD_final, self).__init__()
        self.conv_block_low = nn.Sequential(
            Vanila_Conv_no_pool(in_spatial_low, in_spatial_low // 16, 3),
            nn.Conv2d(in_spatial_low // 16, 1, 1, padding=0),
            nn.Sigmoid()
        )

        self.conv_block_high = nn.Sequential(
            Vanila_Conv_no_pool(in_spatial_high, in_spatial_high // 16, 3),
            nn.Conv2d(in_spatial_high // 16, 1, 1, padding=0),
            nn.Sigmoid()
        )

        self.conv_spatial = nn.Sequential(
            nn.Conv2d(3, 1, 1, padding=0),
            nn.Sigmoid())

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_channel = nn.Sequential(
            nn.Conv2d(in_spatial_low, in_prior, 1, padding=0),
        )
        self.sig = nn.Sigmoid()
        self.Up_to_2 = nn.Upsample(scale_factor=2)

        self.conv_final = Vanila_Conv_no_pool(in_spatial_low, in_prior, 1)
        self.conv_concat = Vanila_Conv_no_pool(in_spatial_low + in_spatial_high, in_spatial_low, 3)

        self.final_attention = nn.Sequential(
            nn.Conv2d(in_prior, in_prior // 4, 3, 1, padding=1),
            nn.BatchNorm2d(in_prior // 4),
            nn.ReLU(),
            nn.Conv2d(in_prior // 4, 1, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x_spatial_low, x_spatial_high):
        b1, c1, w1, h1 = x_spatial_low.size()
        b2, c2, w2, h2 = x_spatial_high.size()
        if (w1, h2) != (w2, h2):
            x_spatial_high = self.Up_to_2(x_spatial_high)

        x_low_map = self.conv_block_low(x_spatial_low)
        x_high_map = self.conv_block_high(x_spatial_high)
        spatial_attention_map = self.conv_spatial(
            torch.cat([x_low_map, x_high_map, Probility_refine(x_low_map, x_high_map)], 1))
        x_spatial_low = self.conv_concat(torch.cat((x_spatial_low, x_spatial_high), 1)) * spatial_attention_map

        channel_attention = self.sig(
            self.conv_channel(self.avg_pool(x_spatial_low)) + self.conv_channel(self.max_pool(x_spatial_low)))

        out = self.conv_final(x_spatial_low) * channel_attention

        return out

class seg_head(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(seg_head, self).__init__()
        self.seg = nn.Sequential(
            Vanila_Conv_no_pool(in_channel, in_channel // 2, 3),
            nn.Conv2d(in_channel // 2, num_classes, 1, 1))
        self.edge = nn.Sequential(
            Vanila_Conv_no_pool(in_channel, in_channel // 2, 3),
            nn.Conv2d(in_channel // 2, num_classes, 1, 1))

    def forward(self, x):
        seg_out = self.seg(x)
        edge_out = self.edge(x)
        return seg_out, edge_out
