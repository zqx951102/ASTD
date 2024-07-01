import numpy as np
import torch
from torch import nn
from torch.nn import init


# https://arxiv.org/abs/2108.01072
def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x
#函数是基于切片和赋值操作来对输入张量进行移位的。它们都通过对不同位置的切片元素进行交换来模拟特定方向的平移。
#之间的区别在于它们的切片操作顺序不同，导致了移动的方向和结果不同。
def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x

# 空间移位"通常指在特征图中进行移位操作。在计算机视觉中，"空间"通常指的是图像或特征图中的位置。这两个函数中的操作通过在特定位置上对张量进行切片和赋值，模拟了不同方向的平移操作。" \
#  "虽然这些操作不是在图像像素级别进行平移，而是通过特征通道的交换来实现，但仍然可以称为"空间移位"，因为它们实质上在特征图的空间位置上进行移动或交换。

class SplitAttention(nn.Module):
    def __init__(self, channel=256, k=3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)
        a = torch.sum(torch.sum(x_all, 1), 1)
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))
        hat_a = hat_a.reshape(b, self.k, c)
        bar_a = self.softmax(hat_a)
        attention = bar_a.unsqueeze(-2)
        out = attention * x_all
        out = torch.sum(out, 1).reshape(b, h, w, c)
        return out


class S2Attention(nn.Module):

    def __init__(self, channels=256):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)  #扩大3倍的通道数
        self.mlp2 = nn.Linear(channels, channels)
        #self.split_attention = SplitAttention()

        # 1x1深度可分离卷积层
        self.dw_conv1 = nn.Conv2d(channels, channels, kernel_size=1, groups=channels)
        self.dw_conv2 = nn.Conv2d(channels, channels, kernel_size=1, groups=channels)

    def forward(self, x):
        b, c, w, h = x.size()
        x = x.permute(0, 2, 3, 1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:, :, :, :c])      #0-c
        x2 = spatial_shift2(x[:, :, :, c:c * 2]) #c-2c
        x3 = x[:, :, :, c * 2:]                  #2c-3c

        # 进行1x1深度可分离卷积    这些子特征通过深度卷积进行混合后进行融合
        x1 = self.dw_conv1(x1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x2 = self.dw_conv2(x2.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)




       #这种方式是取前c个通道
        # x_all = torch.cat([x1, x2, x3], dim=3)  # 拼接在一起
        # x_all = x_all[:, :, :, :c]  # 降低通道数至c
        # x = self.mlp2(x_all)
        # x = x.permute(0, 3, 1, 2)
        # return x




        # 定义适当的卷积操作来降低通道数  卷积去降低维度
        reducer = nn.Conv2d(in_channels=3 * c, out_channels=c, kernel_size=1).to('cuda')
        x_all = torch.cat([x1, x2, x3], dim=3)  # 拼接在一起
        x_all_reduced = reducer(x_all.permute(0, 3, 1, 2).to('cuda'))  # 降低通道数至c
        x = self.mlp2(x_all_reduced.permute(0, 2, 3, 1))  #它会对降维后的特征进行进一步的变换和提取，以便更好地适应模型的训练目标。
        x = x.permute(0, 3, 1, 2)
        return x





      #注意力的方式去 降低维度
        # x_all = torch.stack([x1, x2, x3], 1)     #堆叠在一起
        # a = self.split_attention(x_all)
        # x = self.mlp2(a)
        # x = x.permute(0, 3, 1, 2)
        # return x



# 这种操作旨在改变特征图中像素或特征点之间的空间位置，从而引入空间变换，以改善模型对于空间不变性和特征学习的鲁棒性。
#
# 在深度学习中，"spatial shift" 操作常用于处理图像、视频等空间数据，以增强模型对于不同位置特征的捕获能力，提高其对于平移、
# 旋转等变换的适应性。这可以通过各种卷积神经网络（CNN）层、注意力机制或自注意力机制来实现。




#空间移位操作可以模拟特征之间的相对位置关系变化，例如，沿着特征通道的不同移位方式可能会导致不同的特征表示。
# 然后，SplitAttention 模块被用于在这些移位后的特征上计算注意力权重，以便聚合这些移位后的特征表示，以期望模型能够更好地关注关键特征并提高性能。

# 特征通道（channel）的移位操作，比如 Shift 操作，通常用于改变特征图内部的通道间关系，但它并不直接针对遮挡问题。在遮挡情况下，部分图像区域被遮挡或缺失，导致部分特征信息丢失，这可能会影响模型的性能。
# Shift 操作本身不能解决遮挡问题，但它有时可以帮助模型对遮挡更具有鲁棒性。通过在特征通道上进行移位操作，它可能有助于模型学习到对于遮挡更具有鲁棒性的特征表示。
# 这种操作的作用在于增加特征的多样性和丰富性，使得模型更可能学习到对部分信息缺失的情况下仍能有效分类的特征表示。但是，它并不是专门为了解决遮挡问题而设计的，而是作为一种正则化或增强模型鲁棒性的方法。
# 要解决遮挡问题，更常见的方法包括但不限于：
# 数据增强：通过旋转、剪裁、缩放、填充等技术增加数据多样性，让模型更好地学习到鲁棒特征。


def count_parameters(model):

#统计一个模型的参数数量

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 测试代码
if __name__ == '__main__':
    # 定义输入
    x = torch.randn(300, 256, 14, 14) #256, 256, 14, 14
    # 定义模型
    model = S2Attention(256)
   # model = MultiScaleConv(256, 256)  #参数少Number of parameters: 350720  使用用这个模块！！！！！！！！！


    # 测试模型
    out = model(x)
    print(out.shape)
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params}")