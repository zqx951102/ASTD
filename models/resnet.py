from collections import OrderedDict
import timm  #se-resnet时候
import pdb
import torch.nn.functional as F
import torchvision
from torch import nn
import torch
import torch.nn as nn
import torch.fft
import torchvision as tv
import torchvision
from functools import reduce
from torchvision import datasets, models, transforms
#from models.cycle_mlp import CycleBlock
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from torch.nn.modules.utils import _pair
import math
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv
#from timm.models import resnet
#from timm.models import resnet
#from models.dla import dla34, dla102
from einops.layers.torch import Rearrange
from models.focal_net import FocalNetBlock  #从这里引入 FocalNetBlock   这两个地方变动最大 需要去研究一下！！
from models.maxvit import MBConv  #从这里引入 MBConv
#from models.MixFormer import MixBlock             #########这个是第一种方案设计Mixblock的

class Vec2Patch(nn.Module):  #将输入的特征向量 转换为图像块（patches）的功能
    def __init__(self, channel, hidden, output_size, kernel_size, stride, padding):
        super(Vec2Patch, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.to_patch = torch.nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, padding=padding) #这个操作会将嵌入的特征向量重组成一个具有指定大小的图像张量
        h, w = output_size

    def forward(self, x):
        feat = self.embedding(x) #通过线性映射将输入的特征向量 从维度 hidden映射到维度c-out
        b, n, c = feat.size()
        feat = feat.permute(0, 2, 1)  #重新排列特征向量的维度，以使其适应 torch.nn.Fold 操作的输入要求
        feat = self.to_patch(feat)

        return feat

####res50时候的
class Backbone(nn.Sequential):
    def __init__(self, resnet):
        super(Backbone, self).__init__()
        #self.model =models.resnet50(pretrained=True)
        self.feature1 = nn.Sequential(resnet.conv1,
                                  resnet.bn1, resnet.relu,resnet.maxpool)
                                  #resnet.layer1,resnet.layer2,resnet.layer3)
        self.layer1= nn.Sequential(resnet.layer1)
        self.layer2= nn.Sequential(resnet.layer2)
        self.layer3= nn.Sequential(resnet.layer3)

        self.out_channels = 1024

    def forward(self, x):
        feat = self.feature1(x)
        layer1=self.layer1(feat)
        layer2=self.layer2(layer1)
        layer3=self.layer3(layer2)

        return OrderedDict([["feat_res4", layer3]])  #输出 res4的特征




# ####### 以下是rensnet50
# class Backbone(nn.Sequential):
#     def __init__(self, resnet):
#         super(Backbone, self).__init__()
#         self.feature1 = nn.Sequential(resnet.conv1,
#                                       resnet.bn1, resnet.act1, resnet.maxpool)
#         self.layer1 = nn.Sequential(resnet.layer1)
#         self.layer2 = nn.Sequential(resnet.layer2)
#         self.layer3 = nn.Sequential(resnet.layer3)
#
#         self.out_channels = 1024
#
#     def forward(self, x):
#         feat = self.feature1(x)
#         layer1 = self.layer1(feat)
#         layer2 = self.layer2(layer1)
#         layer3 = self.layer3(layer2)
#
#         return OrderedDict([["feat_res4", layer3]])

    
class Res5Head(nn.Sequential):
    def __init__(self, resnet):
        super(Res5Head, self).__init__()  # res5
        #self.res5feat=OrderedDict([["layer4", resnet.layer4]])
        #self.layer4 = nn.Sequential(resnet.layer4)
        self.out_channels = [1024, 2048]

        hidden = 256   #中间hidden的数值
        output_size = (14,14)  #希望的特征图的大小 14*14大小
        self.focalNet = FocalNetBlock(dim=hidden, input_resolution=196)  #论文 Focal Modulation Networks

#这是设计的MIxBlock的方法
        # self.MixBlock = MixBlock(dim=hidden, num_heads=8, window_size=7, #num_heads=2可行  4也可行  8也可行
        #                          dwconv_kernel_size=3,
        #                          mlp_ratio=4., qkv_bias=True, qk_scale=None
        #                     )

        self.norm = nn.BatchNorm2d(hidden)   #BN操作

        self.qconv1 = nn.Conv2d(in_channels=1024, out_channels=hidden, kernel_size=1)   #经过res4 后特征大小为1024 所有转换成hidden大小的256
        self.qconv2 = nn.Conv2d(in_channels=hidden, out_channels=1024, kernel_size=1)   #在256升高的1024维度 保持前后一致大小
        #self.mb_conv = MBConv(in_channels=d_model, out_channels=d_model)
        self.patch2vec = nn.Conv2d(1024, hidden, kernel_size=(1,1), stride=(1,1), padding=(0,0))  #通过一个卷积使得1024维度降维到hidden的256维度 卷积大小为1*1
        self.vec2patch = Vec2Patch(1024, hidden, output_size, kernel_size=(1,1), stride=(1,1), padding=(0,0)) #将输入的特征向量 转换为图像块（patches）的功能
        
        self.mbconv = MBConv(hidden, hidden)
        self.final_in = nn.Conv2d(in_channels=1024, out_channels=hidden, kernel_size=1)  #1024维度到hidden的256维度
        self.final_in2 = nn.Conv2d(in_channels=hidden, out_channels=1024, kernel_size=1)  #256到1024维度
        self.final_out = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1)  #1024维度到2048维度
        
                
    def forward(self, x):
        input = x
        b, c, h, w = x.size()  #获得 输入x的 bchw等值
        final_in = self.norm(self.final_in(x))  # 先把输入x从1024维度变成256维度  然后进行BN操作 归一化
        
        #mbconv=self.mbconv(final_in)
        final_in2 = self.final_in2(final_in)  #然后256维度变成1024维度 卷积1*1
        
        trans_feat = self.patch2vec(final_in2)  #再把特征向量1024维度 降维到 256维度

        _, c, h, w = trans_feat.size()  #获取 trans_feat的 c h w信息   _,通常用作占位符，表示忽略该值  也就是求得三个维度 分别是 chw
        trans_feat = trans_feat.view(b, c, -1).permute(0, 2, 1)   #将一个三维的张量 trans_feat 进行重新排列和维度变换     然后变成 第一个维度是b 第二个维度是 c  第三个维度是-1 而 -1 表示将剩余的所有元素展平为一个维度。 也就是h*w的大小 然后permute交换位置 变成 B hw C
        #B hw C

        # 设置 'H' 和 'W' 的值
#######mixblock的方法
        # x_MixBlockfeat = self.MixBlock(trans_feat)  # B*HW*c的输入进入到 focalNet中去 得到x_focal_feat
        # trans_feat = self.vec2patch(x_MixBlockfeat)  + final_in2    #x_focal_feat 进行 转换为图像块（patches）的功能  然后与final_in2  1024维度的特征进行相加



# trans_feat  torch.Size([300, 1024, 14, 14])
#final_out   torch.Size([300, 2048, 14, 14])

        x_focal_feat = self.focalNet(trans_feat)  # B*HW*c的输入进入到 focalNet中去 得到x_focal_feat
        trans_feat = self.vec2patch(x_focal_feat) + final_in2  # x_focal_feat 进行 转换为图像块（patches）的功能  然后与final_in2  1024维度的特征进行相加
        final_out = self.final_out(trans_feat)  #1024维度变成 2048维度

    
        x_feat = F.adaptive_max_pool2d(trans_feat, 1)  # x_feat: torch.Size([300, 1024, 1, 1])


        feat = F.adaptive_max_pool2d(final_out, 1) #feat : torch.Size([300, 2048, 1, 1])
        trans_features = {}  #是一个 OrderedDict
        trans_features["before_trans"] = x_feat
        trans_features["after_trans"] = feat
        return trans_features
    

def build_resnet(name="resnet50", pretrained=True):  #构建resnet50 分别去找 BackBone的特征 和 Res5Head的输出
    from torchvision.models import resnet
    resnet.model_urls["resnet50"] = "https://download.pytorch.org/models/resnet50-f46c3f97.pth"
    #resnet = torchvision.models.resnet.__dict__[name](pretrained=pretrained)
    resnet_model = resnet.resnet50(pretrained=True)

    # freeze layers
    resnet_model.conv1.weight.requires_grad_(False)
    resnet_model.bn1.weight.requires_grad_(False)
    resnet_model.bn1.bias.requires_grad_(False)

    return Backbone(resnet_model), Res5Head(resnet_model)


# #更换backbone为 se-resnet50
# def build_resnet(name="seresnet50", pretrained=True):
#     resnet = timm.create_model(name, pretrained=pretrained)
#
#     # freeze layers
#     resnet.conv1.weight.requires_grad_(False)
#     resnet.bn1.weight.requires_grad_(False)
#     resnet.bn1.bias.requires_grad_(False)
#
#     return Backbone(resnet), Res5Head(resnet)