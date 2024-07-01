# --------------------------------------------------------
# FocalNets -- Focal Modulation Networks
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang (jianwyan@microsoft.com)
# --------------------------------------------------------
import pdb
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
#from timm.data.transforms import _pil_interp
from models.maxvit import MaxViTTransformerBlock, grid_partition, grid_reverse, MBConv, window_partition,window_reverse    #引入的maxvit中的函数
from models.ms_mixing import MixShiftBlock   #在这引入的 Mixshiftblock
from models.smt import Attention
#from models.smt import Block

#from models.s2mlp import S2Attention
from models.shift import ShiftViTBlock
class Mlp(nn.Module):  #MLP代码 线性层+激活函数 dropout 然后再是线性层+激活函数
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)     
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FocalModulation(nn.Module):  #调制函数的代码
    def __init__(self, dim, focal_window, focal_level, focal_factor=2, bias=True, proj_drop=0., use_postln=False):       # #dim=hidden  256  focal_level=3, focal_window=1,
        super().__init__()

        self.dim = dim  #256
        self.focal_window = focal_window  #1
        self.focal_level = focal_level   #3
        self.focal_factor = focal_factor  #2
        self.use_postln = use_postln    #false  #去掉了 原来的   self.use_postln_in_modulation = use_postln_in_modulation self.normalize_modulator = normalize_modulator

        self.attn = Attention(
            dim,   # ca_num_heads=8  mAP 56.2 650G     4 时候 55.7  348G
            ca_num_heads=4, sa_num_heads=8, qkv_bias=False, qk_scale=None,   #ca_num_heads=2 选择用几个head 头  选择4时候需要调制b为2  0.003学习率
            attn_drop=0., proj_drop=0, ca_attention=1,
            expand_ratio=2)

        #self.mix = S2Attention()
        self.mix = ShiftViTBlock(256)


        # self.SAM = Block(dim, ca_num_heads=2, sa_num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        #          use_layerscale=False, layerscale_value=1e-4, drop=0., attn_drop=0.,
        #          drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ca_attention=1, expand_ratio=2)




        self.f = nn.Linear(dim, 2*dim , bias=bias)  #原来 self.f = nn.Linear(dim, 2*dim + (self.focal_level+1), bias=bias)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)

        self.act = nn.GELU()
        self.proj = nn.Linear(2*dim, dim)  #原来 self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()  #这个nn.ModuleList()    就是构建layer的成员变量 ，是一个容器，
        
        self.ms_layers = nn.ModuleList()  #原来无这句话   #这个nn.ModuleList()    就是构建layer的成员变量 ，是一个容器，
                
        self.kernel_sizes = [] #是一个列表 空的
        for k in range(self.focal_level): #循环三次 k从0 开始
            kernel_size = self.focal_factor*k + self.focal_window  #2*K+1  变成 1 3 5 卷积核
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, 
                    groups=dim, padding=kernel_size//2, bias=False),
                    nn.GELU(),
                    )
                )      #这个地方196是这样来的  H, W = self.input_resolution
            self.ms_layers.append(MixShiftBlock(dim, 196, shift_size=3, shift_dist=[-1,1,0], mix_size= [1,3,5]))     #对这个特征进行添加       #原来无这句话
            self.kernel_sizes.append(kernel_size)          
        if self.use_postln:       #原来 if self.use_postln_in_modulation:
            self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, H, W, C)
        """
        C = x.shape[-1]  #提取x向量的最后一个维度信息 就是C 通道数
 #q 变量的 shape torch.Size([300, 256, 14, 14])   ctx torch.Size([300, 256, 14, 14])
        # pre linear projection  线性层
        x = self.f(x).permute(0, 3, 1, 2).contiguous()  #转换成 B C H W     那么 torch.split(x, (C, C), 1) 将返回一个包含两个子张量的元组，每个子张量的形状为 (B, C, H, W)。这实际上是将输入张量沿着通道维度分成了两个相等大小的部分。 命名为 q和ctx
        q, ctx = torch.split(x, (C, C), 1)   #     要分割的输入张量  如果是一个整数,则表示分割成均匀大小的split_size_or_sections个张量 如果是一个列表,则表示分割为指定大小的张量  哪个维度进行分割 原来 q, ctx, self.gates = torch.split(x, (C, C, self.focal_level+1), 1)
        #pdb.set_trace()                                     # torch.split()是一个很有用的函数, 可以轻松地将张量分割成任意形状和大小的张量列表, 以用于后续处理。
        q = q.view(q.size(0), -1, q.size(1))
        q = self.attn(q,14,14)
        q = q.view(q.size(0), 256, 14, 14)

        # q = q.view(q.size(0), -1, q.size(1))
        # q = self.SAM(q, 14, 14)
        # q = q.view(q.size(0), 256, 14, 14)

        ctx = self.mix(ctx)


        # # context aggreation  上下文聚合操作
        # ctx_all = 0
        # for l in range(self.focal_level):            #focal_level为 3  也就是循环了三次的意思
        #     f_ctx = self.focal_layers[l](ctx)    # self.focal_layers = nn.ModuleList()   就是构建layer的成员变量 ，是一个容器，这里面添加是卷积层 1 3 5卷积 且有GElu激活函数
        #     ctx_ms = self.ms_layers[l](f_ctx)    # self.ms_layers = nn.ModuleList() 就是构建layer的成员变量 ，是一个容器，  ctx经过 1 3 5卷积后变成f_ctx 然后在进行ms_layers操作 也就是特征混合打乱的步骤
        #     ctx_all = ctx_all + ctx_ms  #最后进行一个相加融合了 合并成原来的维度了 C
        # ctx_global = self.act(f_ctx.mean(2, keepdim=True).mean(3, keepdim=True))  # self.act = nn.GELU() 另外加上了一个 global的特征  全局池化的信息。 这行代码的目的是对输入的特征张量 f_ctx 进行全局平均池化（在高度和宽度上取平均），然后将结果传递给激活函数 self.act。这个过程通常用于在深度神经网络中引入全局上下文信息，以帮助网络更好地理解输入数据的整体特征。
        # ctx_all = ctx_all + ctx_global #然后 全部进行融合
        #
        # # focal modulation
        # self.modulator = self.h(ctx_all) # 是一个卷积层  self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)
        # #x_out = torch.cat((q,self.modulator),1)    #  这里吧q和调制后的特征进行了相加    原来 x_out = q*self.modulator


        x_out = torch.cat((q, ctx), 1)  #修改成 q和ctx这个分支相加
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln:  # 原来 if self.use_postln_in_modulation:
            x_out = self.ln(x_out)  #nn.LayerNorm就是一个LN层
        
        # post linear porjection
        x_out = self.proj(x_out)    #最后经历MLP层 self.proj = nn.Linear(2*dim, dim)  #原来 self.proj = nn.Linear(dim, dim)  self.proj_drop = nn.Dropout(proj_drop)
        x_out = self.proj_drop(x_out)
        return x_out

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0

        flops += N * self.dim * (self.dim * 2 + (self.focal_level+1))

        # focal convolution
        for k in range(self.focal_level):
            flops += N * (self.kernel_sizes[k]**2+1) * self.dim

        # global gating
        flops += N * 1 * self.dim 

        #  self.linear
        flops += N * self.dim * (self.dim + 1)

        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class FocalNetBlock(nn.Module):  #主要是引入的这个部分的 block代码  直接把这个块 修改后 引入！！！仔细看看
    r""" Focal Modulation Network Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        focal_level (int): Number of focal levels. 
        focal_window (int): Focal window size at first focal level
        use_layerscale (bool): Whether use layerscale
        layerscale_value (float): Initial layerscale value
        use_postln (bool): Whether use layernorm after modulation
    """
#dim=hidden  256
    def __init__(self, dim, input_resolution, mlp_ratio=4., drop=0., drop_path=0., 
                    act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                    focal_level=3, focal_window=1,
                    use_layerscale=False, layerscale_value=1e-4, 
                    use_postln=False):      #原 去掉了use_postln_in_modulation=False, normalize_modulator=False
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio

        self.focal_window = focal_window
        self.focal_level = focal_level

        self.norm1 = norm_layer(dim)         #这里调制函数等于  FocalModulation  代码往上查看
        self.modulation = FocalModulation(dim, proj_drop=drop, focal_window=focal_window, focal_level=self.focal_level, use_postln=use_postln)  #  原 修改use_postln_in_modulation=use_postln_in_modulation, normalize_modulator=normalize_modulator

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  #根据代码理解其实drop_path就是在一个batch里面随机去除一部分样本  drop_path=0. 输入是啥，直接给输出，不做任何的改变
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0    
        if use_layerscale:  #use_layerscale=False
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

        self.H = 14  #原 None
        self.W = 14  #原 None
        self.mbconv = MBConv(dim, dim)   #MB放在了这

    def forward(self, x):
        H, W = self.H, self.W  #先去获得x特征的H和W维度值 14*14维度
        B, L, C = x.shape      #获得x向量的 B L C值                 #原少下边一句话 shortcut = x
        
        
       
        # MBConv block
        x = x.reshape([B, C,H, W])    #扩展 B C H W ，可以看出 L应该是H*W的值           #原来无这些话
        x = self.mbconv(x)   #然后去应用MBconv卷积 也就是 BN+ dw3*3+SE+conv1*1 外加shoutcut
        
        shortcut = x.view(B, H * W, C)
         # Focal Modulation
        x = self.norm1(shortcut)  #原 x = x if self.use_postln else self.norm1(x)
        # Focal Modulation
        x = x.view(B, H, W, C) #因为modulation函数的输入需要是 B, H, W, C的维度
        x = self.modulation(x).view(B, H * W, C)  #调制函数的代码 调用

        # FFN
        x = shortcut + self.drop_path(self.gamma_1 * x)  #这个是第一个+号的内容 就是shortcut出来的特征与 1*x内容经过dropath相加  这里是恒等映射
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))  #原为    x = x + self.drop_path(self.gamma_2 * (self.norm2(self.mlp(x)) if self.use_postln else self.mlp(self.norm2(x))))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, " \
               f"mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        
        # W-MSA/SW-MSA
        flops += self.modulation.flops(H*W)

        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class BasicLayer(nn.Module):
    """ A basic Focal Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at first focal level
        use_layerscale (bool): Whether use layerscale
        layerscale_value (float): Initial layerscale value
        use_postln (bool): Whether use layernorm after modulation
    """

    def __init__(self, dim, out_dim, input_resolution, depth,
                 mlp_ratio=4., drop=0., drop_path=0., norm_layer=nn.LayerNorm, 
                 downsample=None, use_checkpoint=False, 
                 focal_level=1, focal_window=1, 
                 use_conv_embed=False, 
                 use_layerscale=False, layerscale_value=1e-4, use_postln=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # build blocks
        self.blocks = nn.ModuleList([
            FocalNetBlock(
                dim=dim, 
                input_resolution=input_resolution,
                mlp_ratio=mlp_ratio, 
                drop=drop, 
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                focal_level=focal_level,
                focal_window=focal_window, 
                use_layerscale=use_layerscale, 
                layerscale_value=layerscale_value,
                use_postln=use_postln,    #删掉了 use_postln_in_modulation=use_postln_in_modulation, normalize_modulator=normalize_modulator,
            )
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(
                img_size=input_resolution, 
                patch_size=2, 
                in_chans=dim, 
                embed_dim=out_dim, 
                use_conv_embed=use_conv_embed, 
                norm_layer=norm_layer, 
                is_stem=False
            )
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = x.transpose(1, 2).reshape(x.shape[0], -1, H, W)
            x, Ho, Wo = self.downsample(x)
        else:
            Ho, Wo = H, W        
        return x, Ho, Wo

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=(224, 224), patch_size=4, in_chans=3, embed_dim=96, use_conv_embed=False, norm_layer=None, is_stem=False):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if use_conv_embed:
            # if we choose to use conv embedding, then we treat the stem and non-stem differently
            if is_stem:
                kernel_size = 7; padding = 2; stride = 4
            else:
                kernel_size = 3; padding = 1; stride = 2
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x)        
        H, W = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x, H, W

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class FocalNet(nn.Module):
    r""" Focal Modulation Networks (FocalNets)

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Focal Transformer layer.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False 
        focal_levels (list): How many focal levels at all stages. Note that this excludes the finest-grain level. Default: [1, 1, 1, 1] 
        focal_windows (list): The focal window size at all stages. Default: [7, 5, 3, 1] 
        use_conv_embed (bool): Whether use convolutional embedding. We noted that using convolutional embedding usually improve the performance, but we do not use it by default. Default: False 
        use_layerscale (bool): Whether use layerscale proposed in CaiT. Default: False 
        layerscale_value (float): Value for layer scale. Default: 1e-4 
        use_postln (bool): Whether use layernorm after modulation (it helps stablize training of large models)
    """
    def __init__(self, 
                img_size=224, 
                patch_size=4, 
                in_chans=3, 
                num_classes=1000,
                embed_dim=96, 
                depths=[2, 2, 6, 2], 
                mlp_ratio=4., 
                drop_rate=0., 
                drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, 
                patch_norm=True,
                use_checkpoint=False,                 
                focal_levels=[2, 2, 2, 2], 
                focal_windows=[3, 3, 3, 3], 
                use_conv_embed=False, 
                use_layerscale=False, 
                layerscale_value=1e-4, 
                use_postln=False, 
                **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        embed_dim = [embed_dim * (2 ** i) for i in range(self.num_layers)]

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio
        
        # split image into patches using either non-overlapped embedding or overlapped embedding
        self.patch_embed = PatchEmbed(
            img_size=to_2tuple(img_size), 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim[0], 
            use_conv_embed=use_conv_embed, 
            norm_layer=norm_layer if self.patch_norm else None, 
            is_stem=True)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=embed_dim[i_layer], 
                               out_dim=embed_dim[i_layer+1] if (i_layer < self.num_layers - 1) else None,  
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate, 
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer, 
                               downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                               focal_level=focal_levels[i_layer], 
                               focal_window=focal_windows[i_layer], 
                               use_conv_embed=use_conv_embed,
                               use_checkpoint=use_checkpoint, 
                               use_layerscale=use_layerscale, 
                               layerscale_value=layerscale_value, 
                               use_postln=use_postln,  #删除了use_postln_in_modulation=use_postln_in_modulation, normalize_modulator=normalize_modulator
                    )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {''}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {''}

    def forward_features(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

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

model_urls = {
    "focalnet_tiny_srf": "",
    "focalnet_small_srf": "",
    "focalnet_base_srf": "",
    "focalnet_tiny_lrf": "",
    "focalnet_small_lrf": "",
    "focalnet_base_lrf": "",


    # "focalnet_tiny_srf": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_tiny_srf.pth",
    # "focalnet_tiny_lrf": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_tiny_lrf.pth",
    # "focalnet_small_srf": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_small_srf.pth",
    # "focalnet_small_lrf": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_small_lrf.pth",
    # "focalnet_base_srf": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_base_srf.pth",
    # "focalnet_base_lrf": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_base_lrf.pth",
    # "focalnet_large_fl3": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_large_lrf_384.pth",
    # "focalnet_large_fl4": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_large_lrf_384_fl4.pth",
    # "focalnet_xlarge_fl3": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_xlarge_lrf_384.pth",
    # "focalnet_xlarge_fl4": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_xlarge_lrf_384_fl4.pth",
    # "focalnet_huge_fl3": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_huge_lrf_224.pth",
    # "focalnet_huge_fl4": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_huge_lrf_224_fl4.pth",

}

@register_model
def focalnet_tiny_srf(pretrained=False, **kwargs):
    model = FocalNet(depths=[2, 2, 6, 2], embed_dim=96, **kwargs)
    if pretrained:
        url = model_urls['focalnet_tiny_srf']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def focalnet_small_srf(pretrained=False, **kwargs):
    model = FocalNet(depths=[2, 2, 18, 2], embed_dim=96, **kwargs)
    if pretrained:
        url = model_urls['focalnet_small_srf']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def focalnet_base_srf(pretrained=False, **kwargs):
    model = FocalNet(depths=[2, 2, 18, 2], embed_dim=128, **kwargs)
    if pretrained:
        url = model_urls['focalnet_base_srf']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def focalnet_tiny_lrf(pretrained=False, **kwargs):
    model = FocalNet(depths=[2, 2, 6, 2], embed_dim=96, focal_levels=[3, 3, 3, 3], **kwargs)
    if pretrained:
        url = model_urls['focalnet_tiny_lrf']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def focalnet_small_lrf(pretrained=False, **kwargs):
    model = FocalNet(depths=[2, 2, 18, 2], embed_dim=96, focal_levels=[3, 3, 3, 3], **kwargs)
    if pretrained:
        url = model_urls['focalnet_small_lrf']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def focalnet_base_lrf(pretrained=False, **kwargs):
    model = FocalNet(depths=[2, 2, 18, 2], embed_dim=128, focal_levels=[3, 3, 3, 3], **kwargs)
    if pretrained:
        url = model_urls['focalnet_base_lrf']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def focalnet_tiny_iso_16(pretrained=False, **kwargs):
    model = FocalNet(depths=[12], patch_size=16, embed_dim=192, focal_levels=[3], focal_windows=[3], **kwargs)
    if pretrained:
        url = model_urls['focalnet_tiny_iso_16']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def focalnet_small_iso_16(pretrained=False, **kwargs):
    model = FocalNet(depths=[12], patch_size=16, embed_dim=384, focal_levels=[3], focal_windows=[3], **kwargs)
    if pretrained:
        url = model_urls['focalnet_small_iso_16']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def focalnet_base_iso_16(pretrained=False, **kwargs):
    model = FocalNet(depths=[12], patch_size=16, embed_dim=768, focal_levels=[3], focal_windows=[3], use_layerscale=True, use_postln=True, **kwargs)
    if pretrained:
        url = model_urls['focalnet_base_iso_16']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

if __name__ == '__main__':
    img_size = 224
    x = torch.rand(16, 3, img_size, img_size).cuda()
    # model = FocalNet(depths=[2, 2, 6, 2], embed_dim=96)
    # model = FocalNet(depths=[12], patch_size=16, embed_dim=768, focal_levels=[3], focal_windows=[3], focal_factors=[2])
    model = FocalNet(depths=[2, 2, 6, 2], embed_dim=96, focal_levels=[3, 3, 3, 3]).cuda()
    print(model); model(x)

    flops = model.flops()
    print(f"number of GFLOPs: {flops / 1e9}")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
