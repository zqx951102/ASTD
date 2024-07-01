from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from .swin_transformer import swin_tiny_patch4_window7_224,swin_small_patch4_window7_224,swin_base_patch4_window7_224   #从swin里面引入三种 tiny small base

bonenum = 3

class Backbone(nn.Sequential):
    def __init__(self, swin, out_channels=384):
        super().__init__()
        self.swin = swin
        self.out_channels = out_channels

    def forward(self, x):
        if self.swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0],1) * self.swin.semantic_weight
            w = torch.cat([w, 1-w], axis=-1)
            semantic_weight = w.cuda()

        x, hw_shape = self.swin.patch_embed(x)

        if self.swin.use_abs_pos_embed:
            x = x + self.swin.absolute_pos_embed
        x = self.swin.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.swin.stages[:bonenum]):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin.semantic_weight >= 0:
                sw = self.swin.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.swin.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.swin.softplus(sw) + sb
            if i == bonenum-1:
                norm_layer = getattr(self.swin, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.swin.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
        return OrderedDict([["feat_res4", outs[-1]]])

class Res5Head(nn.Sequential):
    def __init__(self, swin, out_channels=384):
        super().__init__()  # last block
        self.swin = swin
        self.out_channels = [out_channels, out_channels*2]

    def forward(self, x):
        if self.swin.semantic_weight >= 0:
            w = torch.ones(x.shape[0],1) * self.swin.semantic_weight
            w = torch.cat([w, 1-w], axis=-1)
            semantic_weight = w.cuda()

        feat = x
        hw_shape = x.shape[-2:]
        x = torch.flatten(x, 2)
        x = x.permute(0, 2, 1)
        x,hw_shape = self.swin.stages[bonenum-1].downsample(x,hw_shape)
        if self.swin.semantic_weight >= 0:
            sw = self.swin.semantic_embed_w[bonenum-1](semantic_weight).unsqueeze(1)
            sb = self.swin.semantic_embed_b[bonenum-1](semantic_weight).unsqueeze(1)
            x = x * self.swin.softplus(sw) + sb
        for i, stage in enumerate(self.swin.stages[bonenum:]):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.swin.semantic_weight >= 0:
                sw = self.swin.semantic_embed_w[bonenum+i](semantic_weight).unsqueeze(1)
                sb = self.swin.semantic_embed_b[bonenum+i](semantic_weight).unsqueeze(1)
                x = x * self.swin.softplus(sw) + sb
            if i == len(self.swin.stages) - bonenum - 1:
                norm_layer = getattr(self.swin, f'norm{bonenum+i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.swin.num_features[bonenum+i]).permute(0, 3, 1,
                                                             2).contiguous()
        feat = self.swin.avgpool(feat)
        out = self.swin.avgpool(out)
        return OrderedDict([["feat_res4", feat], ["feat_res5", out]])

def build_swin(name="swin_tiny", semantic_weight=1.0):    #swin_small   swin_base
    if 'tiny' in name:
        swin = swin_tiny_patch4_window7_224(drop_path_rate=0.1,semantic_weight=semantic_weight)
        out_channels = 384
    elif 'small' in name:
        swin = swin_small_patch4_window7_224(drop_path_rate=0.1,semantic_weight=semantic_weight)
        out_channels = 384
    elif 'base' in name:
        swin = swin_base_patch4_window7_224(drop_path_rate=0.1,semantic_weight=semantic_weight)
        out_channels = 512

    return Backbone(swin,out_channels), Res5Head(swin,out_channels), out_channels*2
