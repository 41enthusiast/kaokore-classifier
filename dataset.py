#!/usr/bin/env python3

import csv
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import argparse
import numpy as np
#import lycon
import pandas as pd
#import keras as k
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid
import wandb


def verify_str_arg(value, valid_values):
    assert value in valid_values
    return value


def image_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def load_labels(path):
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        return [{
            headers[column_index]: row[column_index]
            for column_index in range(len(row))
        }
                for row in reader]


class Kaokore(Dataset):

    def __init__(self, root, split='train', category='gender', transform=None):
        self.root = root = os.path.expanduser(root)

        self.split = verify_str_arg(split, ['train', 'dev', 'test'])

        self.category = verify_str_arg(category, ['gender', 'status'])

        self.gen_to_cls = {'male': 0, 'female': 1} if args.label == 'gender' else {'noble': 0, 'warrior': 1, 'incarnation': 2, 'commoner': 3}
        self.cls_to_gen = {v:k for k,v in self.gen_to_cls.items()}

        labels = load_labels(os.path.join(root, 'labels.csv'))
        self.entries = [
            (label_entry['image'], int(label_entry[category]))
            for label_entry in labels
            if label_entry['set'] == split and os.path.exists(
                os.path.join(self.root, 'images_256', label_entry['image']))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        image_filename, label = self.entries[index]

        image_filepath = os.path.join(self.root, 'images_256', image_filename)
        image = image_loader(image_filepath)
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def class_freq_pie(y, paint_labels, cat):
    paint_class_count = torch.bincount(y)
    plt.pie(paint_class_count, labels = paint_labels, autopct='%1.2f%%')
    plt.title(cat+' class frequency of the face cropped dataset')
    plt.savefig('misc/pie_class_freq.jpg')
    plt.clf()
    w_img = wandb.Image('misc/pie_class_freq.jpg', caption='Class Frequency')
    wandb.log({'class_freq':w_img})


def color_hist(y, lbls, dataset, workers):
    #get subsets of the dataset classes for the color histogram analysis
    classes_indices = [(y==i).nonzero(as_tuple = True)[0] for i in lbls.values()]

    for cls_indices, class_name in zip(classes_indices, lbls.keys()):
        cls_subset = Subset(dataset, cls_indices)
        cls_loader = DataLoader(
            cls_subset,
            num_workers=workers,
            batch_size=len(cls_subset),
        )
        c_x, _ = next(iter(cls_loader))

        #color histogram

        imhist_r = torch.histc(c_x[0,:,:], bins = 255, min = 0, max =1)
        imhist_g = torch.histc(c_x[1,:,:], bins = 255, min = 0, max =1)
        imhist_b = torch.histc(c_x[2,:,:], bins = 255, min = 0, max =1)
        plt.bar(range(0,255), imhist_r, align = 'center', color = 'r', alpha = 0.4)
        plt.bar(range(0,255), imhist_g, align = 'center', color = 'g', alpha = 0.4)
        plt.bar(range(0,255), imhist_b, align = 'center', color = 'b', alpha = 0.4)
        plt.title('Color histogram for the class '+ class_name)
        plt.savefig('misc/'+class_name+'_color_hist.jpg')
        plt.clf()
        w_img = wandb.Image('misc/'+class_name+'_color_hist.jpg', caption='Color Histogram')
        wandb.log({'color_hist': w_img})

    


#dataset analysis
if __name__=='__main__':
    
    #models = {'vgg16': VGG16, 'resnet50': ResNet50, 'mobilenetv2': MobileNetV2, 'densenet121': DenseNet121}


    parser = argparse.ArgumentParser(description="Train a Keras model on the KaoKore dataset")
    #parser.add_argument('--arch', type=str, choices=models.keys(), required=True)
    parser.add_argument('--label', type=str, choices=['gender', 'status'], required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)

    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='Number of workers (default: 4)')
    #parser.add_argument('--epochs', type=int, default=20)
    #parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam')
    #arser.add_argument('--lr-adjust-freq' , type=int, default=10, help='How many epochs per LR adjustment (*=0.1)')

    #parser.add_argument('--lr', type=float, default=0.001)
    #parser.add_argument('--momentum', type=float, default=0.9)
    #parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (L2 penalty)')

    args = parser.parse_args()

    wandb.init(project='kaokore-dataset-analysis', name=args.version,
     config = {'type': args.label, 'image-size': args.image_size})

    df = pd.read_csv(f'{args.root}/labels.csv')
    image_dir = f'{args.root}/images_256/'

    train_transforms = transforms.Compose([transforms.ToTensor(),
     transforms.Resize((args.image_size, args.image_size))])

    train_ds = Kaokore(args.root, 'train', args.label, transform=train_transforms)
    val_ds = Kaokore(args.root, 'dev', args.label)
    test_ds = Kaokore(args.root, 'test', args.label)

    print('Total images in the train dataset: ',len(train_ds))
    print('Total images in the validation dataset: ',len(val_ds))
    print('Total images in the test dataset: ',len(test_ds))
    
    #training dataset analysis
    train_loader = DataLoader(train_ds, batch_size=len(train_ds), num_workers= args.num_workers, shuffle=True)
    x, y = next(iter(train_loader)) 

    #class freq analysis
    print('Logging the class frequency')
    class_freq_pie(y, train_ds.gen_to_cls.keys(), args.label)

    print('Logging a sample image grid')
    img_grid = make_grid(x[:16], 4)
    w_img = wandb.Image(img_grid, caption='Image grid')
    wandb.log({'img_grid': w_img})

    print('Logging per class color histograms')
    color_hist(y, train_ds.gen_to_cls, train_ds, args.num_workers)


    print('Program finished.')


