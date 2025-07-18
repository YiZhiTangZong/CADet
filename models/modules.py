import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mrf3 import FFD

# MRF3 的多路卷积
class MPE(nn.Module):
    def __init__(self, in_channel, use_gau=True, reduce_dim=False, out_channels=None):
        super(MPE, self).__init__()
        print("已加载MPE")
        self.Conv_1 = nn.Sequential(
            nn.Conv2d(in_channel // 4, in_channel // 4, 1, 1, padding=0),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )

        self.Conv_3 = nn.Sequential(
            nn.Conv2d(in_channel // 4, in_channel // 4, 3, 1, padding=1),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )

        self.Conv_5 = nn.Sequential(
            nn.Conv2d(in_channel // 4, in_channel // 4, 5, 1, padding=2),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )

        self.Conv_7 = nn.Sequential(
            nn.Conv2d(in_channel // 4, in_channel // 4, 7, 1, padding=3),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )

        self.Conv = Vanila_Conv_no_pool(in_channel * 2, in_channel, 1)

        self.se = SE_Block(in_channel)

        # 后续处理模块 --------------------------------------------------
        # GAU输入通道数调整为C（降维后）
        self.gau = GAU(
            in_channels = in_channel,  # 关键修改点
            use_gau = use_gau,
            reduce_dim = reduce_dim,
            out_channels = out_channels if out_channels else in_channel
        )

    def forward(self, x, y):
        b, c, h, w = x.size()
        x_1 = x[:, :(c // 4), :, :]
        x_2 = x[:, (c // 4):(c // 4) * 2, :, :]
        x_3 = x[:, (c // 4) * 2:(c // 4) * 3, :, :]
        x_4 = x[:, (c // 4) * 3:, :, :]

        x_4_7 = self.Conv_7(x_4)
        x_3_5 = self.Conv_5(x_3)
        x_2_3 = self.Conv_3(x_2)
        x_1_1 = self.Conv_7(x_1)

        out = self.se(self.Conv(torch.cat((x_1_1, x_2_3, x_3_5, x_4_7, x), 1)))

        return self.gau(out, y)

# 自注意力
class SAT(nn.Module):
    def __init__(self, in_channels, sat_pos, num_layers=2, num_heads=2, use_gau=True, reduce_dim=False, out_channels=None):
        super(SAT, self).__init__()

        print("已加载SAT, 大小", sat_pos)
        
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        self.num_layers = num_layers
        self.self_attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.MultiheadAttention(in_channels, num_heads),
                'norm': nn.LayerNorm(in_channels),
                'activation': nn.ReLU()
            })
            for _ in range(num_layers)
        ])
        
        # 可学习的位置编码（示例，需根据输入尺寸调整）
        self.pos_embedding = nn.Parameter(torch.randn(1, in_channels, sat_pos, sat_pos))

        # 后续处理模块 --------------------------------------------------
        # GAU输入通道数调整为C（降维后）
        self.gau = GAU(
            in_channels = in_channels,  # 关键修改点
            use_gau = use_gau,
            reduce_dim = reduce_dim,
            out_channels = out_channels if out_channels else in_channels
        )

    def forward(self, x, y):
        batch_size, channels, height, width = x.shape
        # 添加位置编码
        pos_embed = F.interpolate(self.pos_embedding, size=(height, width), mode='bilinear', align_corners=False)
        x = x + pos_embed
        
        x_seq = x.view(batch_size, channels, -1).permute(2, 0, 1)  # [seq_len, batch, channels]
        
        for layer in self.self_attention_layers:
            residual = x_seq
            attn_output, _ = layer['attention'](x_seq, x_seq, x_seq)
            attn_output += residual  # 残差连接
            attn_output = layer['norm'](attn_output)
            x_seq = layer['activation'](attn_output)
        
        attn_output = x_seq.permute(1, 2, 0).view(batch_size, channels, height, width)

        return self.gau(attn_output, y)
    

# 多路卷积
class MCC(nn.Module):
    def __init__(self, in_channels, use_gau=True, reduce_dim=False, out_channels=None):
        """
        改进版多路卷积模块（四路并行）
        
        Args:
            in_channels (int): 输入/输出通道数
            use_gau (bool): 是否使用GAU模块
            reduce_dim (bool): 是否降维
            out_channels (int): 输出通道数（当reduce_dim=True时生效）
        """
        super(MCC, self).__init__()

        print("已加载MCC")

        # 四路分支定义 --------------------------------------------------
        # 分支0: 恒等映射（直接保留原始特征）
        self.identity = nn.Identity()
        
        # 分支1: 3x3卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        
        # 分支2: 5x5卷积
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        
        # 分支3: 7x7卷积
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        # 降维层：将通道数从 4 * C 降为 C
        self.reduce = nn.Sequential(
            nn.Conv2d(4 * in_channels, in_channels, kernel_size=1),  # 1x1卷积降维
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        # 后续处理模块 --------------------------------------------------
        # GAU输入通道数调整为C（降维后）
        self.gau = GAU(
            in_channels = in_channels,  # 关键修改点
            use_gau = use_gau,
            reduce_dim = reduce_dim,
            out_channels = out_channels if out_channels else in_channels
        )

    def forward(self, x, y):
        # 四路并行计算
        b0 = self.identity(x)  # [B, C, H, W]
        b3 = self.conv3(x)     # [B, C, H, W]
        b5 = self.conv5(x)     # [B, C, H, W]
        b7 = self.conv7(x)     # [B, C, H, W]
        
        # 通道维度合并（concat后为[B, 4C, H, W]）
        concat = torch.cat([b0, b3, b5, b7], dim=1)
        
        # 降维到 [B, C, H, W]
        reduced = self.reduce(concat)
        
        # 通过GAU处理
        return self.gau(reduced, y)  # 输出维度由GAU决定
    

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv2d(1, 1, kernel_size=(5, 5), padding=(2, 2)),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.activate = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out = self.fc2(self.fc1(self.max_pool(x)))
        out = avg_out + max_out
        out = self.activate(out)
        return out

class MergeSigmoid(nn.Module):
    def __init__(self):
        super(MergeSigmoid, self).__init__()
        print(f"合并器启用MergeSigmoid, 直接数学融合")
        
    def forward(self, x1, x2):
        # 输入x1和x2已经是sigmoid后的结果
        a = x1
        b = x2
        
        # 计算融合公式：ab/(ab + (1-a)(1-b))
        numerator = a * b
        denominator = numerator + (1-a)*(1-b)
        
        # 添加极小值防止除零
        out = numerator / (denominator + 1e-6)
        return out
    
class Merge(nn.Module):
    def __init__(self, channel):
        super(Merge, self).__init__()
        print(f"合并器启用Merge, 卷积通道降维")
        self.dual_merge = nn.Sequential(    # 2C合并为C
            nn.Conv2d(channel, channel // 2, kernel_size=1),
            nn.BatchNorm2d(channel // 2),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        out = self.dual_merge(x)  # 进行通道合并
        return out
     
class MergeCA(nn.Module):
    def __init__(self, channel, reduction=2):
        super(MergeCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(channel, channel // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(channel // reduction, channel, 1, bias=False)
        self.activate = nn.Sigmoid()
        self.dual_merge = nn.Sequential(    # 2C合并为C
            nn.Conv2d(channel, channel // 2, kernel_size=1),
            nn.BatchNorm2d(channel // 2),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out = self.fc2(self.fc1(self.max_pool(x)))
        out = avg_out + max_out
        out = self.activate(out)
        out = self.dual_merge(out)  # 进行通道合并
        return out

class GAU(nn.Module):
    def __init__(self, in_channels, use_gau=True, reduce_dim=False, out_channels=None):
        super(GAU, self).__init__()
        self.use_gau = use_gau
        if self.use_gau:
            print("已加载GAU")
        self.reduce_dim = reduce_dim

        if self.reduce_dim:
            self.down_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            in_channels = out_channels

        if self.use_gau:

            self.sa = SpatialAttention()
            self.ca = ChannelAttention(in_channels)

            self.reset_gate = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, y):
        if self.reduce_dim:
            x = self.down_conv(x)

        if self.use_gau:
            y = F.interpolate(y, x.shape[-2:], mode='bilinear', align_corners=True)

            comx = x * y
            resx = x * (1 - y) # bs, c, h, w

            x_sa = self.sa(resx) # bs, 1, h, w
            x_ca = self.ca(resx) # bs, c, 1, 1

            O = self.reset_gate(comx)
            M = x_sa * x_ca

            RF = M * x + (1 - M) * O
        else:
            RF = x
        return RF

class FIM(nn.Module):
    def __init__(self, in_channels, out_channels, f_channels, use_topo=True, up=True, use_fusion=True, bottom=False):
        super(FIM, self).__init__()
        self.use_topo = use_topo
        self.up = up
        self.bottom = bottom
        self.use_fusion = use_fusion

        if self.up:
            self.up_s = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.up_t = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.up_s = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.up_t = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.decoder_s = nn.Sequential(
            nn.Conv2d(out_channels + f_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.inner_s = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        if self.bottom:
            self.st = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

        if self.use_topo:
            self.decoder_t = nn.Sequential(
                nn.Conv2d(out_channels + out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            self.s_to_t = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            self.t_to_s = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            self.res_s = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            self.inner_t = nn.Sequential(
                nn.Conv2d(out_channels, 1, kernel_size=3, padding=1, bias=False),
                nn.Sigmoid()
            )

        if self.use_fusion:
            print("FIM启用feature fusion")
            self.fusion = FFD(in_spatial_low=in_channels // 2, in_spatial_high=in_channels // 2, in_prior=in_channels)

    def forward(self, x_s, x_t, rf):
        if self.use_topo:
            if self.bottom:
                x_t = self.st(x_t)
            #bs, c, h, w = x_s.shape
            x_s = self.up_s(x_s)
            x_t = self.up_t(x_t)

            # padding
            diffY = rf.size()[2] - x_s.size()[2]
            diffX = rf.size()[3] - x_s.size()[3]

            x_s = F.pad(x_s, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            x_t = F.pad(x_t, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

            # 这是原本的cat的特征融合方法，对这里下手
            if self.use_fusion:
                rf_s = self.fusion(rf, x_s) # def forward(self, x_spatial_low, x_spatial_high)，第一个参数是低层特征（分辨率高），第二个参数是高层特征（分辨率低）
            else:
                rf_s = torch.cat((x_s, rf), dim=1)
            
            s = self.decoder_s(rf_s)
            s_t = self.s_to_t(s)

            t = torch.cat((x_t, s_t), dim=1)
            x_t = self.decoder_t(t)
            t_s = self.t_to_s(x_t)

            s_res = self.res_s(torch.cat((s, t_s), dim=1))

            x_s = s + s_res
            t_cls = self.inner_t(x_t)
            s_cls = self.inner_s(x_s)
        else:
            x_s = self.up_s(x_s)
            #x_b = self.up_b(x_b)
            # padding
            diffY = rf.size()[2] - x_s.size()[2]
            diffX = rf.size()[3] - x_s.size()[3]

            x_s = F.pad(x_s, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])

            rf_s = torch.cat((x_s, rf), dim=1)
            s = self.decoder_s(rf_s)
            x_s = s
            x_t = x_s
            t_cls = None
            s_cls = self.inner_s(x_s)
        return x_s, x_t, s_cls, t_cls

class BaseDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, f_channels):
        super(BaseDecoder, self).__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels, f_channels, kernel_size=2, stride=2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(f_channels * 2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, rf):

        x = self.up_conv(x, output_size=rf.size())

        #padding
        diffY = rf.size()[2] - x.size()[2]
        diffX = rf.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        y = self.conv1(torch.cat([x, rf], dim=1))
        y = self.conv2(y)

        return y

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








