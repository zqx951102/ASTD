import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math

from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
#from timm.data.transforms import _pil_interp
from ptflops import get_model_complexity_info
#from thop import profile


class Mlp(nn.Module):  # MLP感知器
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
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
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):  # 交叉注意力 ca_attention=1
    def __init__(self, dim, ca_num_heads=4, sa_num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., ca_attention=1, expand_ratio=2):
        super().__init__()

        self.ca_attention = ca_attention  # 是0或者1 判断
        self.dim = dim
        self.ca_num_heads = ca_num_heads  # ca是4
        self.sa_num_heads = sa_num_heads  # sa是8
       #方法中增加门控机制相关的参数
        # self.gating_weights = nn.Parameter(torch.ones(self.ca_num_heads))  # 定义门控权重参数，初始化为全1

        assert dim % ca_num_heads == 0, f"dim {dim} should be divided by num_heads {ca_num_heads}."  # 也就是能整除吧
        assert dim % sa_num_heads == 0, f"dim {dim} should be divided by num_heads {sa_num_heads}."

        self.act = nn.GELU()  # 激活
        self.proj = nn.Linear(dim, dim)  # 线性层
        self.proj_drop = nn.Dropout(proj_drop)  # drop层

        self.split_groups = self.dim // ca_num_heads  # 维度的切分 通道

        if ca_attention == 1:
            self.v = nn.Linear(dim, dim, bias=qkv_bias)  # 这个地方其实就是生成KV
            self.s = nn.Linear(dim, dim, bias=qkv_bias)
            for i in range(self.ca_num_heads):  # 也就是循环四次  然后去做卷积操作  卷积大小是 从3+2i  padding也在变  一共有dim//self.ca_num_heads维度吧 没一组的通道数
                # kernel_size = 2 * i + 1
                # padding = i
                # ##修改成 1 3 5 7递增的方式      # 创建 nn.Conv2d 层
                # local_conv = nn.Conv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=kernel_size,
                #                        padding=padding, stride=1, groups=dim // self.ca_num_heads)
                # setattr(self, f"local_conv_{i + 1}", local_conv)
               ##下面代码是 3 5 7 9这种递增方式
                local_conv = nn.Conv2d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=(3 + i * 2),  # 更换k的大小 kernel_size=(3 + i * 2),
                                       padding=(1 + i), stride=1, groups=dim // self.ca_num_heads)
                setattr(self, f"local_conv_{i + 1}", local_conv)
            # 这行代码的作用是将当前对象self的属性名设置为动态生成的字符串f"local_conv_{i + 1}"，并将其值设置为local_conv。
            # 这样可以在对象上动态地创建属性，属性名是由字符串和变量i的值组成的，使得属性名具有一定的动态性和可变性。
            self.proj0 = nn.Conv2d(dim, dim * expand_ratio, kernel_size=1, padding=0, stride=1,
                                   groups=self.split_groups)
            self.bn = nn.BatchNorm2d(dim * expand_ratio)  # 先升高维度在去降低维度到dim
            self.proj1 = nn.Conv2d(dim * expand_ratio, dim, kernel_size=1, padding=0, stride=1)

        else:  # 自注意力（ca_attention=0）
            head_dim = dim // sa_num_heads
            self.scale = qk_scale or head_dim ** -0.5  # 一个因子进行缩放
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

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
        B, N, C = x.shape
        if self.ca_attention == 1:
            v = self.v(x)  # 两个线性层操作
            s = self.s(x).reshape(B, H, W, self.ca_num_heads, C // self.ca_num_heads).permute(3, 0, 4, 1, 2)
            for i in range(self.ca_num_heads):  # 这是一个循环，遍历了通道注意力的每个头部。
                local_conv = getattr(self, f"local_conv_{i + 1}")  # 这行代码获取了当前迭代的局部卷积操作（local_conv_i）
                s_i = s[i]  # 这里提取了通道注意力头 i 对应的张量。
                s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, W)  # 取的张量 s_i 应用了局部卷积操作，然后进行了形状重塑，以便后续的拼接。
                # #使用门控机制 根据门控权重调整每个尺度卷积头的权重
                # #这样做可以使得每个尺度卷积头的输出被门控机制调节，根据门控权重的不同值，动态地调整不同尺度卷积头的重要性，使模块能够自适应地选择适合当前情况的尺度信息。
                # gating_weight = torch.sigmoid(self.gating_weights[i])  # 计算当前头部的门控权重，使用 sigmoid 确保权重在 (0, 1) 范围内
                # s_i *= gating_weight  # 使用门控权重调整当前头部的输出张量


                if i == 0:
                    s_out = s_i
                else:  # 每个通道注意力头 s_i 生成整体的注意力张量 s_out。
                    s_out = torch.cat([s_out, s_i], 2)  # 否则，将新的 s_i 与之前的 s_out 进行拼接（torch.cat 操作），沿着第三个维度（通道维度）进行拼接。
            s_out = s_out.reshape(B, C, H, W)  # 进行形状重塑，以匹配后续的操作
            s_out = self.proj1(self.act(self.bn(self.proj0(s_out))))  # s-out进行升高维度 Bn 再去降低维度到 dim
            self.modulator = s_out
            s_out = s_out.reshape(B, C, N).permute(0, 2, 1)  # 调整形状 BCN BNC
            x = s_out * v  # v的shape B, N, C    sout也是BNC   注意力矩阵和值矩阵，它们经过处理后应该具有相同的维度，以便能够进行逐元素相乘操作。
        # 在这种模式下（ca_attention = 1），模块执行一种交叉注意力。它通过同时进行空间注意力和通道注意力机制来处理输入x。输入x通过线性变换v和s
        # 进行处理，然后用于计算标记 / 特征之间的注意力机制。对于空间注意力，代码根据头数（ca_num_heads）将输入分成多个组，并对每个组应用单独的卷积操作（local_conv），然后重新整形和连接结果。
        # 对连接的空间注意力输出应用调制步骤（proj0、bn、proj1）。最后，将调制后的注意力与通道变换后的输入v进行逐元素乘法。
        else:
            q = self.q(x).reshape(B, N, self.sa_num_heads, C // self.sa_num_heads).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, -1, 2, self.sa_num_heads, C // self.sa_num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + \
                self.local_conv(v.transpose(1, 2).reshape(B, N, C).transpose(1, 2).view(B, C, H, W)).view(B, C,
                                                                                                          N).transpose(
                    1, 2)
        # 注意力分数通过矩阵乘法计算，并通过一个因子进行缩放（self.scale）。然后应用了一个dropout。
        # 输出根据注意力分数和值张量计算得出。此外，在将值张量融入输出之前，对值张量应用了局部卷积操作。
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):  #这个 Block 模块可以看作是一个由多头注意力机制和多层感知机组成的特征处理单元，用于在 Transformer 架构中实现特征提取和转换。

    def __init__(self, dim, ca_num_heads, sa_num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 use_layerscale=False, layerscale_value=1e-4, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ca_attention=1, expand_ratio=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            ca_num_heads=ca_num_heads, sa_num_heads=sa_num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, ca_attention=ca_attention,
            expand_ratio=expand_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

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

        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=3, stride=2, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        img_size = to_2tuple(img_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

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


class Head(nn.Module):
    def __init__(self, head_conv, dim):
        super(Head, self).__init__()
        stem = [nn.Conv2d(3, dim, head_conv, 2, padding=3 if head_conv == 7 else 1, bias=False), nn.BatchNorm2d(dim),
                nn.ReLU(True)]
        stem.append(nn.Conv2d(dim, dim, kernel_size=2, stride=2))
        self.conv = nn.Sequential(*stem)
        self.norm = nn.LayerNorm(dim)
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
        x = self.conv(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class SMT(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16], mlp_ratios=[8, 6, 4, 2],
                 qkv_bias=False, qk_scale=None, use_layerscale=False, layerscale_value=1e-4, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[2, 2, 8, 1], ca_attentions=[1, 1, 1, 0], num_stages=4, head_conv=3, expand_ratio=2, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = Head(head_conv, embed_dims[i])  #
            else:
                patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                                patch_size=3,
                                                stride=2,
                                                in_chans=embed_dims[i - 1],
                                                embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], ca_num_heads=ca_num_heads[i], sa_num_heads=sa_num_heads[i], mlp_ratio=mlp_ratios[i],
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                ca_attention=0 if i == 2 and j % 2 != 0 else ca_attentions[i], expand_ratio=expand_ratio)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

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

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def build_transforms(img_size, center_crop=False):
    t = []
    if center_crop:
        size = int((256 / 224) * img_size)
        t.append(
            transforms.Resize(size, interpolation=_pil_interp('bicubic'))
        )
        t.append(
            transforms.CenterCrop(img_size)
        )
    else:
        t.append(
            transforms.Resize(img_size, interpolation=_pil_interp('bicubic'))
        )
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_transforms4display(img_size, center_crop=False):
    t = []
    if center_crop:
        size = int((256 / 224) * img_size)
        t.append(
            transforms.Resize(size, interpolation=_pil_interp('bicubic'))
        )
        t.append(
            transforms.CenterCrop(img_size)
        )
    else:
        t.append(
            transforms.Resize(img_size, interpolation=_pil_interp('bicubic'))
        )
    t.append(transforms.ToTensor())
    return transforms.Compose(t)


def smt_t(pretrained=False, **kwargs):
    model = SMT(
        embed_dims=[64, 128, 256, 512], ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16],
        mlp_ratios=[4, 4, 4, 2],
        qkv_bias=True, depths=[2, 2, 8, 1], ca_attentions=[1, 1, 1, 0], head_conv=3, expand_ratio=2, **kwargs)
    model.default_cfg = _cfg()

    return model


def smt_s(pretrained=False, **kwargs):
    model = SMT(
        embed_dims=[64, 128, 256, 512], ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16],
        mlp_ratios=[4, 4, 4, 2],
        qkv_bias=True, depths=[3, 4, 18, 2], ca_attentions=[1, 1, 1, 0], head_conv=3, expand_ratio=2, **kwargs)
    model.default_cfg = _cfg()

    return model


def smt_b(pretrained=False, **kwargs):
    model = SMT(
        embed_dims=[64, 128, 256, 512], ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16],
        mlp_ratios=[8, 6, 4, 2],
        qkv_bias=True, depths=[4, 6, 28, 2], ca_attentions=[1, 1, 1, 0], head_conv=7, expand_ratio=2, **kwargs)
    model.default_cfg = _cfg()

    return model


def smt_l(pretrained=False, **kwargs):
    model = SMT(
        embed_dims=[96, 192, 384, 768], ca_num_heads=[4, 4, 4, -1], sa_num_heads=[-1, -1, 8, 16],
        mlp_ratios=[8, 6, 4, 2],
        qkv_bias=True, depths=[4, 6, 28, 4], ca_attentions=[1, 1, 1, 0], head_conv=7, expand_ratio=2, **kwargs)
    model.default_cfg = _cfg()

    return model


if __name__ == '__main__':
    import torch

    model = smt_t()
    model = model.cuda()
    # input = torch.rand(4, 3, 224, 224).cuda()
    # output = model(input)
    print(model)

    ### thop cal ###
    # input_shape = (1, 3, 384, 384) # 输入的形状
    # input_data = torch.randn(*input_shape)
    # macs, params = profile(model, inputs=(input_data,))
    # print(f"FLOPS: {macs / 1e9:.2f}G")
    # print(f"params: {params / 1e6:.2f}M")

    ### ptflops cal ###
    flops_count, params_count = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                                          print_per_layer_stat=False)

    print('flops: ', flops_count)
    print('params: ', params_count)