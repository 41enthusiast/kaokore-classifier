import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import cv2
from torchvision.models import vgg16, densenet121, resnet50

import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

from dataset import Kaokore, gen_val_transforms, gen_train_transforms


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


def vgg_model(pooling='avg', freeze_base=True):
    model = vgg16(True)

    if pooling == 'avg':
        for i, layer in model.features.named_children():
            if isinstance(layer, torch.nn.MaxPool2d):
                model.features[int(i)] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    if freeze_base:
        print("Freezing embeddings")
        for param in model.parameters():
            param.requires_grad = False

    return retrieve_all_conv_layers(model.features)


def resnet_model(pooling='avg', freeze_base=True):
    model = resnet50(True)

    if pooling == 'avg':
        model.maxpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    if freeze_base:
        print("Freezing embeddings")
        for param in model.parameters():
            param.requires_grad = False

    # to get the features only
    #del model.fc
    #del model.avgpool

    return retrieve_all_conv_layers(model)


def densenet_model(pooling='avg', freeze_base=True):
    model = densenet121(True)

    if pooling == 'avg':
        for c, (i, layer) in enumerate(model.features.named_children()):
            if isinstance(layer, torch.nn.MaxPool2d):
                model.features[c] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    if freeze_base:
        print("Freezing embeddings")
        for param in model.parameters():
            param.requires_grad = False

    return retrieve_all_conv_layers(model.features)


def get_conv_part_of_models(arch, pooling='avg'):
    models = {'vgg': vgg_model, 'resnet': resnet_model, 'densenet': densenet_model}

    return models[arch](pooling)


def retrieve_all_conv_layers(model):

    model_weights = []  # we will save the conv layer weights in this list
    conv_layers = []  # we will save the 49 conv layers in this list

    # get all the model children as list
    model_children = list(model.children())

    # counter to keep count of the conv layers
    counter = 0
    # append all the conv layers and their respective weights to the list
    for j in range(len(model_children)):
        if type(model_children[j]) == nn.Conv2d:
            counter += 1
            model_weights.append(model_children[j].weight)
            conv_layers.append(model_children[j])
        elif type(model_children[j]) == nn.Sequential:
            for k in range(len(model_children[j])):
                for child in model_children[j][k].children():
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolutional layers: {counter}")
    return model_weights, conv_layers


def visualize_model_conv_filters(model_weights):
    print('Visualizing the first conv layer filters')
    plt.figure(figsize=(20, 17))
    for i, filter in enumerate(model_weights[0]):
        plt.subplot(8, 8, i + 1)  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(filter[0, :, :].detach(), cmap='gray')
        plt.axis('off')
        plt.savefig('misc/model_viz/filter.png')

    w_img = wandb.Image('misc//model_viz/filter.png', caption='First Conv layer filters')
    wandb.log({'Model_Conv_Filters': w_img})
    #plt.show()
    plt.close()


def visualize_feat_maps(conv_layers, img):
    # pass the image through all the layers
    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))
    # make a copy of the `results`
    outputs = results
    print('Results shapes:')
    for i in outputs:
        print(i.shape)#bsz, op_filter, img_h, img_w

    print('Visualizing 64 features from each layer')
    # (although there are more feature maps in the upper layers)
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(30, 30))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data#num_filters, img_h, img_w
        for i, filter in enumerate(layer_viz):
            if i == 64:  # we will visualize only 8x8 blocks from each layer
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter, cmap='gray')
            plt.axis("off")
        print(f"Saving layer {num_layer} feature maps...")
        plt.savefig(f"misc/model_viz/layer_{num_layer}.png")
        # plt.show()
        w_img = wandb.Image(f'misc/model_viz/layer_{num_layer}.png', caption=f'Layer {num_layer}')
        wandb.log({'Model_Feature_Map': w_img})
        plt.close()


def get_attention_info(x, model):

    logits, att_mat = model(x.unsqueeze(0))

    att_mat = torch.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    grid_size = int(np.sqrt(aug_att_mat.size(-1)))

    return joint_attentions, grid_size


def visualize_attention_maps(joint_att, grid_size, img, img_size)
    for i, v in enumerate(joint_att):
        v = joint_att[-1]
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), img_size)[..., np.newaxis]
        result = (mask * img).astype("uint8")

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
        ax1.set_title('Original')
        ax2.set_title('Attention Map_%d Layer' % (i + 1))
        _ = ax1.imshow(img)
        _ = ax2.imshow(result)
        plt.savefig(f"misc/model_viz/attn_layer_{i+1}.png")
        # plt.show()
        w_img = wandb.Image(f'misc/model_viz/layer_{i+1}.png', caption=f'Attention on Layer {i+1}')
        wandb.log({'Attention on Layer': w_img})


# model visualization
if __name__ == '__main__':
    model_choices = ['vgg', 'resnet', 'densenet']

    parser = argparse.ArgumentParser(description="Train a Keras model on the KaoKore dataset")
    parser.add_argument('--arch', type=str, choices=model_choices, required=True)
    parser.add_argument('--label', type=str, choices=['gender', 'status'], required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)

    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='Number of workers (default: 4)')

    # parser.add_argument('--epochs', type=int, default=20)
    # parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam')
    # arser.add_argument('--lr-adjust-freq' , type=int, default=10, help='How many epochs per LR adjustment (*=0.1)')

    # parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (L2 penalty)')

    args = parser.parse_args()

    wandb.init(project='kaokore-model-visualization', name=args.version,
               config={'type': args.label, 'image-size': args.image_size})

    # Set up dataset
    df = pd.read_csv(f'{args.root}/labels.csv')
    image_dir = f'{args.root}/images_256/'

    train_ds = Kaokore(args, 'train', args.label, transform=gen_val_transforms(args))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    x, y = next(iter(train_loader))

    img_grid = make_grid(x, 4)
    w_img = wandb.Image(img_grid, caption='Image grid')
    wandb.log({'img_grid': w_img})

    x0 = x[0].unsqueeze(0)
    w_img = wandb.Image(x0, caption=f'Input Image')
    wandb.log({'input_image': w_img})

    model_wts, conv_lyrs = get_conv_part_of_models(args.arch)(x)

    visualize_feat_maps(conv_lyrs, x0)
    visualize_model_conv_filters(model_wts)

    # Attention

    #z = get_conv_part_of_models(args.arch)(x)
    #q = torch.reshape(z, (z.size(0), -1 , z.size(1)))# bsz, seqlen, feat

    print(z.shape, q.shape, x.shape, y.shape)

    query = torch.rand(128, 32, 1, 256)
    key = value = torch.rand(128, 16, 1, 256)
    query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    multihead_attn = ScaledDotProductAttention(temperature=query.size(2))
    attn_output, attn_weights = multihead_attn(query, key, value)
    attn_output = attn_output.transpose(1, 2)
    print(f'attn_output: {attn_output.size()}, attn_weights: {attn_weights.size()}')

    for i, v in enumerate(joint_att1):
        v = joint_att1[-1]
        mask = v[0, 1:].reshape(grid_size1, grid_size1).detach().numpy()
        mask = cv2.resize(mask / mask.max(), img1.size)[..., np.newaxis]
        result = (mask * img1).astype("uint8")

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))
        ax1.set_title('Original')
        ax2.set_title('Attention Map_%d Layer' % (i + 1))
        _ = ax1.imshow(img1)
        _ = ax2.imshow(result)
