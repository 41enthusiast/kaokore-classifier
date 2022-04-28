import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import ProjectorBlock, SpatialAttn
from torchvision import models
from collections import OrderedDict
import math

"""
VGG-16 with attention
"""
class AttnVGG(nn.Module): #the vgg n densnet versions
    def __init__(self, num_classes, output_layers, attention=True, normalize_attn=True):
        super(AttnVGG, self).__init__()
        # conv blocks
        self.pretrained = models.vgg16(True).features
        self.fhooks = []
        self.selected_out = OrderedDict()
        self.dense = nn.AdaptiveAvgPool2d((1,1))
        #self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(sample_size / 32), padding=0,
        #                       bias=True)

        for i, l in enumerate(list(self.pretrained._modules.keys())):
            if i in output_layers:
                self.fhooks.append(getattr(self.pretrained, l).register_forward_hook(self.forward_hook(l)))

        # attention blocks
        self.attention = attention
        if self.attention:
            self.projector0 = ProjectorBlock(64, 512)
            self.projector1 = ProjectorBlock(128, 512)

            self.attn0 = SpatialAttn(in_features=512, normalize_attn=normalize_attn)# (batch_size,1,H,W), (batch_size,C)
            self.attn1 = SpatialAttn(in_features=512, normalize_attn=normalize_attn)
            self.attn2 = SpatialAttn(in_features=512, normalize_attn=normalize_attn)
            self.attn3 = SpatialAttn(in_features=512, normalize_attn=normalize_attn)

        # final classification layer
        if self.attention:
            self.classify = nn.Linear(in_features=512*4, out_features=num_classes, bias=True)
        else:
            self.classify = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x):
        out = self.pretrained(x)
        l0, l1, l2, l3 = self.selected_out.values()
        g = self.dense(out) # batch_sizex512x1x1
        #print(g.shape, l0.shape, l1.shape, l2.shape, l3.shape)
        # attention
        if self.attention:
            c0, g0 = self.attn0(self.projector0(l0), g)
            c1, g1 = self.attn1(self.projector1(l1), g)#this gets it to the same out ch as the next 2 layers
            c2, g2 = self.attn2(l2, g)
            c3, g3 = self.attn3(l3, g)
            g = torch.cat((g0, g1,g2,g3), dim=1) # batch_sizex3C
            # classification layer
            out = self.classify(g) # batch_sizexnum_classes
        else:
            c0, c1, c2, c3 = None, None, None
            out = self.classify(torch.squeeze(g))
        return [out, c0, c1, c2, c3]






# Test
if __name__ == '__main__':
    model = AttnVGG(num_classes=10, output_layers=[0, 7, 21, 28])
    x = torch.randn(16,3,256,256)
    out, c0, c1, c2, c3 = model(x)
    print(out.shape, c0.shape, c1.shape, c2.shape, c3.shape)