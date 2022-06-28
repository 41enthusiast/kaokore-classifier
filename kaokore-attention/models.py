import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import ProjectorBlock, SpatialAttn
from torchvision import models
from collections import OrderedDict
import math

from utils import drop_connect

"""
VGG-16 with attention
"""
class AttnVGG(nn.Module): #the vgg n densnet versions
    def __init__(self, num_classes, output_layers, dropout_mode, p, attention=True, normalize_attn=True):
        super(AttnVGG, self).__init__()
        # conv blocks
        self.pretrained = models.vgg16(True).features

        # Freeze Parameters
        for param in self.pretrained.parameters():
            param.requires_grad = False

        self.fhooks = []
        self.selected_out = OrderedDict()
        self.dense = nn.AdaptiveAvgPool2d((1, 1))
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

        #dropout selection for type of regularization
        self.dropout_mode, self.p = dropout_mode, p

        if self.dropout_mode == 'dropout':
            self.dropout = nn.Dropout(self.p)
        elif self.dropout_mode == 'dropconnect':
            self.dropout = drop_connect

        # final classification layer
        if self.attention:
            self.fc1 = nn.Linear(in_features=512*4, out_features=512, bias=True)
            self.classify = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        else:
            self.fc1 = nn.Linear(in_features=512, out_features=256, bias=True)
            self.classify = nn.Linear(in_features=256, out_features=num_classes, bias=True)

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

            # fc layer
            out = torch.relu(self.fc1(g)) # batch_sizexnum_classes
            #print(out.shape)

            if self.dropout_mode == 'dropout':
                out = self.dropout(out)
            elif self.dropout_mode == 'dropconnect':
                out = self.dropout(out, self.p, self.training)
            #print(out.shape)

        else:
            c0, c1, c2, c3 = None, None, None
            out = self.fc1(torch.squeeze(g))

            if self.dropout_mode == 'dropout':
                out = self.dropout(out)
            elif self.dropout_mode == 'dropconnect':
                out = self.dropout(out, self.p, self.training)

        out = self.classify(out)
        return [out, c0, c1, c2, c3]



class AttnResnet(nn.Module): #the vgg n densnet versions
    def __init__(self, num_classes, output_layers, dropout_mode, p, attention=True, normalize_attn=True):
        super(AttnResnet, self).__init__()
        # conv blocks
        self.pretrained = models.resnet50(True)

        # Freeze Parameters
        #for param in self.pretrained.parameters():
        #    param.requires_grad = False

        self.fhooks = []
        self.selected_out = OrderedDict()
        #self.pretrained.avgpool = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, padding=0,
        #                       bias=True)
        self.dense = nn.Conv2d(in_channels=2048, out_channels=512, kernel_size=1, padding=0,
                               bias=True)

        for i in output_layers:
            lyr = None
            lyr_name = []
            temp_model = self.pretrained
            m_list = list(temp_model._modules.keys())
            for j in i.split('.')[:-1]:
                lyr = getattr(temp_model, m_list[int(j)])
                lyr_name.append(m_list[int(j)])
                temp_model = lyr
                m_list = list(temp_model._modules.keys())
            lyr = getattr(temp_model, m_list[int(i.split('.')[-1])])
            lyr_name.append(m_list[int(i.split('.')[-1])])
            self.fhooks.append(lyr.register_forward_hook(self.forward_hook('_'.join(lyr_name))))
        self.fhooks.append(self.pretrained.avgpool.register_forward_hook(self.forward_hook('out')))

        # attention blocks
        self.attention = attention
        if self.attention:
            self.projector0 = ProjectorBlock(64, 512)
            self.projector1 = ProjectorBlock(256, 512)
            self.projector2 = ProjectorBlock(256, 512)

            self.attn0 = SpatialAttn(in_features=512, normalize_attn=normalize_attn)# (batch_size,1,H,W), (batch_size,C)
            self.attn1 = SpatialAttn(in_features=512, normalize_attn=normalize_attn)
            self.attn2 = SpatialAttn(in_features=512, normalize_attn=normalize_attn)
            self.attn3 = SpatialAttn(in_features=512, normalize_attn=normalize_attn)

        # dropout selection for type of regularization
        self.dropout_mode, self.p = dropout_mode, p

        if self.dropout_mode == 'dropout':
            self.dropout = nn.Dropout(self.p)
        elif self.dropout_mode == 'dropconnect':
            self.dropout = drop_connect

        # final classification layer
        if self.attention:
            self.fc1 = nn.Linear(in_features=512 * 4, out_features=512, bias=True)
            self.classify = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        else:
            self.fc1 = nn.Linear(in_features=512, out_features=256, bias=True)
            self.classify = nn.Linear(in_features=256, out_features=num_classes, bias=True)


    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x):
        out = self.pretrained(x)
        l0, l1, l2, l3, out = self.selected_out.values()
        g =self.dense(out) # batch_sizex512x1x1
        #print(g.shape, l0.shape, l1.shape, l2.shape, l3.shape)

        # attention
        if self.attention:
            c0, g0 = self.attn0(self.projector0(l0), g)
            c1, g1 = self.attn1(self.projector1(l1), g)#this gets it to the same out ch as the next 2 layers
            c2, g2 = self.attn2(self.projector2(l2), g)
            c3, g3 = self.attn3(l3, g)
            g = torch.cat((g0, g1,g2,g3), dim=1) # batch_sizex3C

            # fc layer
            out = torch.relu(self.fc1(g))  # batch_sizexnum_classes
            # print(out.shape)

            if self.dropout_mode == 'dropout':
                out = self.dropout(out)
            elif self.dropout_mode == 'dropconnect':
                out = self.dropout(out, self.p, self.training)

            # classification layer
            out = self.classify(out) # batch_sizexnum_classes
        else:
            c0, c1, c2, c3 = None, None, None



            if self.dropout_mode == 'dropout':
                out = self.dropout(out)
            elif self.dropout_mode == 'dropconnect':
                out = self.dropout(out, self.p, self.training)

            out = self.classify(torch.squeeze(g))
        return [out, c0, c1, c2, c3]



# Test
if __name__ == '__main__':
    model = AttnVGG(num_classes=10, output_layers=[0, 7, 21, 28], dropout_mode='dropconnect', p=0.2)
    x = torch.randn(16,3,256,256)
    out, c0, c1, c2, c3 = model(x)
    print('VGG', out.shape, c0.shape, c1.shape, c2.shape, c3.shape)

    model = AttnResnet(num_classes=10, output_layers=['0', '4.1.4', '6.2.2', '7.1.2'], dropout_mode='dropout', p=0.2)
    x = torch.randn(16, 3, 256, 256)
    out, c0, c1, c2, c3 = model(x)
    print('Resnet', out.shape, c0.shape, c1.shape, c2.shape, c3.shape)