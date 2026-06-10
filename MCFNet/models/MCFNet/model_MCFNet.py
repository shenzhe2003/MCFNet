import torch
import math
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch.fft
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg
import math
import numpy as np
from einops import rearrange, repeat, einsum
from mamba_ssm import Mamba
from .Modules.DCC import DCC as dcc
from .Modules.MS_FFN import *

def Loss(pred, target):
    return SoftIoULoss(pred, target)+bce_loss(pred, target) 

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state = 64, expand = 2, d_conv = 4, conv_bias = True,  bias = False ):
        super().__init__()
        self.d_model = d_model
        self.d_state=d_state
        self.d_conv=d_conv
        self.expand=expand
        self.conv_bias=conv_bias
        self.bias=bias
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=self.bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=self.conv_bias,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=self.bias)
        
    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)
        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        output = self.out_proj(y)
        return output
    
    def ssm(self, x):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan2d(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    def selective_scan2d(self, u, delta, A, B, C, D, use=True):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        if not use:
            try:
                h = w = int(l ** 0.5)
                assert h * w == l, "Sequence length must be square for SS2D"
                u_2d = u.view(b, h, w, d_in)
                deltaA_2d = deltaA.view(b, h, w, d_in, n)
                deltaB_u_2d = deltaB_u.view(b, h, w, d_in, n)
                C_2d = C.view(b, h, w, n)
                x_h = torch.zeros((b, w, d_in, n), device=u.device)
                h_outs = []
                for i in range(h):
                    x_h = deltaA_2d[:, i] * x_h + deltaB_u_2d[:, i]
                    out_h = einsum(x_h, C_2d[:, i], 'b w d_in n, b w n -> b w d_in')
                    h_outs.append(out_h)
                h_outs = torch.stack(h_outs, dim=1)
                x_w = torch.zeros((b, h, d_in, n), device=u.device)
                w_outs = []
                for j in range(w):
                    x_w = deltaA_2d[:, :, j] * x_w + deltaB_u_2d[:, :, j]
                    out_w = einsum(x_w, C_2d[:, :, j], 'b h d_in n, b h n -> b h d_in')
                    w_outs.append(out_w)
                w_outs = torch.stack(w_outs, dim=2)
                y = h_outs + w_outs
                y = y + u_2d * D
                y = y.view(b, l, d_in)
                return y
            except:
                use = True
        if use:
            x = torch.zeros((b, d_in, n), device=deltaA.device)
            ys = []
            for i in range(l):
                x = deltaA[:, i] * x + deltaB_u[:, i]
                y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
                ys.append(y)
            y = torch.stack(ys, dim=1)
            y = y + u * D
            return y

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=64, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
    def forward(self, x):
        B, L, C = x.shape
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)    
        return x_mamba

class MCMB(nn.Module):
    def __init__(self, 
        dim, 
        mlp_ratio,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        sr_ratio=1, 
        cm_type = 'MSFFN' 
    ):
        super().__init__()
        self.norm2 = norm_layer(dim)
        self.attn = MambaLayer(dim)
        self.mlp = MSFFN(in_features=dim, hidden_features=int(dim * mlp_ratio))       
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class DownSamples(nn.Module):
    def __init__(self, in_channels, out_channels, is_down=True):
        super().__init__()
        if not is_down:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class glo_bran(nn.Module):
    def __init__(self, 
        in_chans=3, 
        num_classes=2, 
        stem_hidden_dim = 32,
        embed_dims=[64, 64, 128, 256],
        mlp_ratios=[8, 8, 4, 4], 
        drop_path_rate=0., 
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3], 
        sr_ratios=[4, 2, 1, 1], 
        num_stages=4,
        token_label=True,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        alpha=5
        for i in range(num_stages):
            if i == 0:                
                patch_embed = Stem(in_chans, stem_hidden_dim, embed_dims[i])
            elif i == 1:
                patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i], is_down=False)
            else:
                patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i])
            block = nn.ModuleList([MCMB(
                dim = embed_dims[i], 
                mlp_ratio = mlp_ratios[i], 
                drop_path=dpr[cur + j], 
                norm_layer=norm_layer,
                sr_ratio = sr_ratios[i],
                cm_type='MSFFN')
            for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)    
        self.return_dense = token_label
        self.mix_token = token_label
        self.beta = 1.0
        self.pooling_scale = 8
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_tokens(self, x, H, W):    
        B = x.shape[0]
        x = x.view(B, -1, x.size(-1))

        for i in range(self.num_stages):
            if i != 0:
                patch_embed = getattr(self, f"patch_embed{i + 1}")
                x, H, W = patch_embed(x)
            block = getattr(self, f"block{i + 1}")
            for blk in block:
                x = blk(x, H, W)
            if i != self.num_stages - 1:
                norm = getattr(self, f"norm{i + 1}")
                x = norm(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        norm = getattr(self, f"norm{self.num_stages}")
        x = norm(x)    
        return x

    def forward_embeddings(self, x):
        patch_embed = getattr(self, f"patch_embed{0 + 1}")
        x, H, W = patch_embed(x)
        x = x.view(x.size(0), H, W, -1)
        return x, H, W

    def forward(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            x, H, W = patch_embed(x)
            block = getattr(self, f"block{i + 1}")
            for blk in block:
                x = blk(x, H, W)
            if i != self.num_stages - 1:
                norm = getattr(self, f"norm{i + 1}")
                x = norm(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                xx = x.reshape(B, H*4, W*4, -1).permute(0, 3, 1, 2).contiguous()
                outs.append(xx)
        norm = getattr(self, f"norm{self.num_stages}")
        x = norm(x) 
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        xx = x.reshape(B, H*4, W*4, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(xx)           
        return outs
            
class FFDM(nn.Module):
    def __init__(self, channels, r=4):
        super(FFDM, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // r)
        self.topdown_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.channels, out_channels=self.bottleneck_channels, kernel_size=7, padding=7 // 2, bias=False),
            nn.BatchNorm2d(self.bottleneck_channels, momentum=0.9),
            nn.Conv2d(in_channels=self.bottleneck_channels, out_channels=self.channels, kernel_size=1, stride=1,padding=0),
            nn.BatchNorm2d(self.channels, momentum=0.9),
            nn.Sigmoid()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.topdown_2 = nn.Sequential(
            nn.Linear(channels, channels // 16, bias=False),  # 降维
            nn.ReLU(inplace=True),  # 激活
            nn.Linear(channels // 16, channels, bias=False),  # 恢复通道数
            nn.Sigmoid()  # 通道注意力权重
        )
        self.bottomup_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.bottleneck_channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(self.bottleneck_channels, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.bottleneck_channels, out_channels=self.channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(self.channels, momentum=0.9),
            nn.Sigmoid()
        )
        self.bottomup_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.bottleneck_channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(self.bottleneck_channels, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.bottleneck_channels, out_channels=self.channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(self.channels, momentum=0.9),
            nn.Sigmoid()
        )
        self.post_1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.ReLU(inplace=True)
        )
        self.post_2 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.ReLU(inplace=True)
        )
        self.post = nn.Sequential(
            nn.BatchNorm2d(channels, momentum=0.9),
            nn.ReLU(inplace=True)
        )

    def forward(self, xh, xl_1, xl_2):
        b, c, _, _ = xh.size()
        # 通道感知的滤波
        xh_temp2 = self.avg_pool(xh).view(b, c)
        topdown_wei_2 = self.topdown_2(xh_temp2).view(b, c, 1, 1)
        # 空间激活
        max_out, _ = torch.max(xh, dim=1, keepdim=True)
        max_out = max_out.expand(-1, c//2, -1, -1)
        avg_out = torch.mean(xh, dim=1, keepdim=True)
        avg_out = avg_out.expand(-1, c//2, -1, -1)
        xh_temp1 = torch.cat([max_out, avg_out], dim=1)
        topdown_wei_1 = self.topdown_1(xh_temp1)
        bottomup_wei_1 = self.bottomup_1(xl_1)
        bottomup_wei_2 = self.bottomup_2(xl_2)
        xs_1 = 2 * torch.mul(xl_1, topdown_wei_1) + 2 * torch.mul(xh, bottomup_wei_1)
        xs_2 = 2 * torch.mul(xl_2, topdown_wei_2) + 2 * torch.mul(xh, bottomup_wei_2)
        xs = self.post_1(xs_1) + self.post_1(xs_2)
        xs = self.post(xs)
        return xs

class InitialFus(nn.Module):
    def __init__(self, in_ch):
        super(InitialFus, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch), 
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch*2),
            nn.ReLU(inplace=True)
        )
        self.se = SELayer(in_ch*2)  
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch*2, in_ch*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch*2),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_ch*2, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, e):
        e1 = self.conv1(e)
        out = self.conv3(e1)
        ca = self.se(out)
        out = self.conv4(out + ca)
        out = self.conv5(out)
        return out
    
class DCCB(nn.Module):
    expansion: int = 1
    def __init__(
            self,
            inplanes: int,
            planes: int,
            downsample=None,
            norm_layer=None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = dcc(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        if out.shape[1] == identity.shape[1] and out.shape[2] == identity.shape[2]:
            out += identity
        out = self.relu(out)
        return out

class MCFNet(nn.Module):
    def __init__(self, block=BasicBlock, deep_supervision=False, inplanes=[3, 16, 32, 64, 128, 256], layers=[2, 2, 2, 2], scale=0.1):
        super(MCFNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.scale = scale 
        self.RshapeSize = nn.ModuleList() 
        for i in range(4):
            if i == 0 :
                self.RshapeSize.append(BasicBlock(4, 16))
            else:
                self.RshapeSize.append(BasicBlock(2**(i+1), 2**(i+1)*8))
        self.layer_0 = self._make_layer(DECBlock, inplanes[0], inplanes[1], 1, stride=1)
        self.layer_1 = self._make_layer(DECBlock, inplanes[1], inplanes[2], layers[0], stride=1)
        self.layer_2 = self._make_layer(DECBlock, inplanes[2], inplanes[3], layers[1], stride=1)
        self.layer_3 = self._make_layer(DECBlock, inplanes[3], inplanes[4], layers[2], stride=1)
        self.encoder3 = glo_bran(
        stem_hidden_dim = 64,
        embed_dims = [64, 64, 128, 256], 
        mlp_ratios = [0.125, 0.125, 0.125, 0.125],
        norm_layer = partial(nn.LayerNorm, eps=1e-6), 
        depths = [3, 4, 12, 3], 
        sr_ratios = [4, 2, 1, 1])
        self.encoder3.default_cfg = _cfg()
        self.initial_fus = InitialFus(inplanes[5])
        self.layer_4_0 = self._make_layer(block, inplanes[4] * 2, inplanes[5], layers[3], stride=1)
        self.layer_3_1 = self._make_layer(block, inplanes[5] + 2 * inplanes[4], inplanes[4], layers[2], stride=1)
        self.layer_2_1 = self._make_layer(block, inplanes[4] + 2 * inplanes[3], inplanes[3], layers[2], stride=1)
        self.layer_1_1 = self._make_layer(block, inplanes[3] + 2 * inplanes[2], inplanes[2], layers[2], stride=1)
        self.layer_0_1 = self._make_layer(block, inplanes[2] + 2 * inplanes[1], inplanes[1], 1, stride=1)
        self.BConv3 = BasicBlock(128, 64)
        self.BConv2 = BasicBlock(64, 32)
        self.BConv1 = BasicBlock(32, 16)
        self.fuse_0 = FFDM(inplanes[1])
        self.fuse_1 = FFDM(inplanes[2])
        self.fuse_2 = FFDM(inplanes[3])
        self.fuse_3 = FFDM(inplanes[4])
        self.layer_f_0 = nn.Conv2d(inplanes[1], 1, kernel_size=1)
        self.layer_f_1 = nn.Conv2d(inplanes[2], 1, kernel_size=1)
        self.layer_f_2 = nn.Conv2d(inplanes[3], 1, kernel_size=1)
        self.layer_f_3 = nn.Conv2d(inplanes[4], 1, kernel_size=1)
        self.layer_f_4 = nn.Conv2d(inplanes[5], 1, kernel_size=1)
        self.final_layer = nn.Conv2d(5, 1, kernel_size=1)

    def forward(self, input, input1, warm_epoch=False):
        x_0 = self.layer_0(input)                            
        x_1 = self.layer_1(x_0)                             
        x_2 = self.layer_2(self.pool(x_1))                   
        x_3 = self.layer_3(self.pool(x_2))                   
        outs = self.encoder3(input1)
        out0 = self.RshapeSize[0](outs[0])
        out1 = self.RshapeSize[1](outs[1])
        out2 = self.RshapeSize[2](outs[2])
        out3 = self.RshapeSize[3](outs[3])
        x_4_0 = self.layer_4_0(torch.cat([self.pool(x_3), self.pool(out3)], 1))
        mul = self.initial_fus(x_4_0)
        x_3_1 = self.layer_3_1(torch.cat([x_3, out3, self.up(x_4_0)], 1))
        x_2_1 = self.layer_2_1(torch.cat([x_2, out2, self.up(x_3_1)], 1))
        x_1_1 = self.layer_1_1(torch.cat([x_1, out1, self.up(x_2_1)], 1))
        x_0_1 = self.layer_0_1(torch.cat([x_0, out0, x_1_1], 1))
        fuse3 = self.fuse_3(x_3_1, x_3, out3)
        fuse2 = self.fuse_2(self.relu(self.up(self.BConv3(fuse3))*self.scale+x_2_1), x_2, out2)
        fuse1 = self.fuse_1(self.relu(self.up(self.BConv2(fuse2))*self.scale+x_1_1), x_1, out1)
        fuse0 = self.fuse_0(self.relu(self.BConv1(fuse1)*self.scale+x_0_1), x_0, out0)     
        f_0_0 = self.layer_f_0(self.relu(fuse0 + x_0_1))
        f_0_1 = self.layer_f_1(self.relu(fuse1 + x_1_1))
        f_0_2 = self.layer_f_2(self.up(self.relu(fuse2 + x_2_1)))
        f_0_3 = self.layer_f_3(self.up_4(self.relu(fuse3 + x_3_1)))
        f_0_4 = self.layer_f_4(self.up_8(mul))

        if warm_epoch:
            final = self.final_layer(torch.cat([f_0_0, f_0_1, f_0_2, f_0_3, f_0_4], 1))
            return [f_0_0, f_0_1, f_0_2, f_0_3, f_0_4], final
        else:
            final = self.final_layer(torch.cat([f_0_0, f_0_1, f_0_2, f_0_3, f_0_4], 1))
            return final
