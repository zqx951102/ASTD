import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
######   (ref) https://github.com/JegZheng/MS-MLP/blob/main/models/ms_mlp.py
###ECCV22-TokenMix 参考一下学习

class MixShiftBlock(nn.Module):   #引入的这个block 进行查看  来自论文：Mixing and Shifting: Exploiting Global and Local Dependencies in Vision MLPs
    r""" Mix-Shifting Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        shift_size (int): Shift size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    #dim 256  input_resolution 196  shift_size=3, shift_dist=[-1,1,0], mix_size= [1,3,5]

    def __init__(self, dim, input_resolution, shift_size, shift_dist, mix_size,  layer_scale_init_value=1e-6,
                 mlp_ratio=4, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio

        self.shift_size = shift_size  #3 是用于定义混合位移的尺寸
        self.shift_dist = shift_dist  #shift_dist=[-1,1,0] 是混合位移的距离。
        self.chunk_size = [i.shape[0] for i in torch.chunk(torch.zeros(dim), self.shift_size)]  #总的来说，这段代码的目的是获取分割操作后每个子块的大小，并将这些大小存储在 self.chunk_size 列表中，以便后续的操作使用。这在一些需要知道子块大小的情况下非常有用，
#使用 torch.chunk 函数将上述全零张量在指定的维度上分割成多个子块。具体来说，它将全零张量按照维度 dim（这个维度由 dim 决定）分割成 self.shift_size 个子块。
#它的作用是将全零张量按照指定维度 dim 分割成多个子块，并将这些子块的大小存储在self.chunk_size列表中供后续使用。这对于需要知道子块大小的操作非常有用。
        self.dwDConv_lr =  nn.Conv2d(dim, dim, kernel_size=1, stride=1, groups=dim, padding=0, bias=False)  #DW卷积
        self.dwDConv_td =  nn.Conv2d(dim, dim, kernel_size=1, stride=1, groups=dim, padding=0, bias=False)  #DW卷积
# 是混合位移块中使用的深度可分离卷积层，用于对特征进行水平和垂直的位移操作。这里使用了深度可分离卷积，即深度卷积和逐点卷积的组合，
# 以减少参数数量和计算量。其中dwDConv_lr 是用于左右移动的卷积层，dwDConv_td是用于上下移动的卷积层。
        ##和源代码的区别地方 少了这么多
        # self.dwconv_lr = nn.ModuleList(
        #     [nn.Conv2d(chunk_dim, chunk_dim, kernel_size=kernel_size[0], padding=kernel_size[1], groups=chunk_dim) for
        #      chunk_dim, kernel_size in zip(self.chunk_size, self.kernel_size)])
        # self.dwconv_td = nn.ModuleList(
        #     [nn.Conv2d(chunk_dim, chunk_dim, kernel_size=kernel_size[0], padding=kernel_size[1], groups=chunk_dim) for
        #      chunk_dim, kernel_size in zip(self.chunk_size, self.kernel_size)])
        #
        # self.norm = LayerNorm(dim, eps=1e-6)
        # self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))  # pointwise/1x1 convs, implemented with linear layers
        # self.act = nn.GELU()
        # self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
        #                           requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#第二组代码中的 forward 函数则是对输入张量进行了分组和复制，然后对复制的张量进行了零填充以实现移位。它首先将输入张量分成多个子组，
# 然后根据指定的移位距离，在特定维度上创建副本，并用零进行填充。这里的移位是通过复制并填充零值来模拟的，
# 类似于一种手动的移位操作。在这个过程中，分别对左右和上下的特征进行了移位和填充操作，最后将移位后的特征进行了卷积并相加。
    def forward(self, x):
        input = x
        B_, C, H, W = x.shape

        # split groups
        xs = torch.chunk(x, self.shift_size, 1)   #这段代码的目的是获取分割操作后每个子块的大小，  3
        # tmp
        x_shift_lr=[]
        for x_c, shift in zip(xs, self.shift_dist):  #创建左右和上下平移的副本：对于每个子组 x_c 和对应的平移 shift，创建副本 x_c_f。如果平移是1（即不进行平移），则副本保持不变。否则，通过将副本的特定维度上的某些元素替换为零来执行平移操作。
            x_c_f = x_c.clone()  #副本
            if shift == 1:
               x_c_f = x_c_f
            else:               #通过将副本的特定维度上的某些元素替换为零来执行平移操作。
                t_a=  x_c_f[:,:,:,shift]  #对于左右平移（水平方向），在通道维度上进行平移：即对于 x_c_f 的第三维度（在通道维度上），如果 shift 不等于1，则将 x_c_f 的右侧某一列设置为零。
                z_ta = torch.zeros(t_a.shape)
                x_c_f[:,:,:,shift] = z_ta
            x_shift_lr.append(x_c_f)
        
        x_shift_td=[]
        for x_c, shift in zip(xs, self.shift_dist):
            x_c_f = x_c.clone()  #副本
            if shift == 1:
               x_c_f = x_c_f
            else:               #通过将副本的特定维度上的某些元素替换为零来执行平移操作。
                t_a=  x_c_f[:,:,shift,:]  #对于上下平移（垂直方向），在高度维度上进行平移：即对于 x_c_f 的第二维度（在高度维度上），如果 shift 不等于1，则将 x_c_f 的底部某一行设置为零。
                z_ta = torch.zeros(t_a.shape)
                x_c_f[:,:,shift,:] = z_ta
            x_shift_td.append(x_c_f)

       ###原始代码是这样  上面进行了修改
        #  # shift with pre-defined relative distance
        # x_shift_lr = [ torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, self.shift_dist)]
        # x_shift_td = [ torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, self.shift_dist)]

        # # regional mixing
        # for i in range(self.shift_size):
        #     x_shift_lr[i] = self.dwconv_lr[i](x_shift_lr[i])
        #     x_shift_td[i] = self.dwconv_td[i](x_shift_td[i])
        
        x_lr = torch.cat(x_shift_lr, 1)  #水平和垂直方向上的平移合并：将经过水平方向平移和垂直方向平移后的副本 x_shift_lr 和 x_shift_td 进行拼接，分别得到 x_lr 和 x_td。
        x_td = torch.cat(x_shift_td, 1)
        
        x_lr = self.dwDConv_lr(x_lr)  #执行深度可分离卷积：通过 self.dwDConv_lr 和 self.dwDConv_td 对 x_lr 和 x_td 进行深度可分离卷积操作。
        x_td = self.dwDConv_td(x_td)
        
        x = x_lr + x_td  #将平移后的结果相加：将水平方向和垂直方向上的结果 x_lr 和 x_td 相加，得到最终的输出张量 x。
        #在增强混合之间存在一个残差连接，从而产生针对部分遮挡的稳健表示。
        x = input + self.drop_path(x) #通过 self.drop_path 函数对输出 x 执行丢弃路径操作，这是一种正则化技术，有助于减少过拟合。
        return x  #总体来说，这段代码实现了一个模型中的一层操作，其中包括通道的分割、平移操作、深度可分离卷积、结果相加和丢弃路径等步骤。这种操作通常用于深度神经网络中，以提取特征并学习数据的表示。

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, " \
               f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        N = H * W
        # dwconv_1 dwconv_2
        for i in range(self.shift_size):
            flops += 2 * (N * self.chunk_size[i] * self.kernel_size[i][0])
        # x_lr + x_td
        flops += N * self.dim
        # norm
        flops += self.dim * H * W
        # pwconv
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        return flops

