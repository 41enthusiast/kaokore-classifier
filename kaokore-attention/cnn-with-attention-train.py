import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils
from torchvision.datasets import ImageFolder
from tensorboardX import SummaryWriter
import os
import argparse
import numpy as np
from datetime import datetime
from models import AttnVGG, AttnResnet
from functions import train_epoch, val_epoch, visualize_attn

from dataset import Kaokore
from utils import focal_loss, make_confusion_matrix, plot_confusion_matrix, get_most_and_least_confident_predictions

hyperparameter_defaults = dict(
  dataset_path='../../kaokore_imagenet_style/status/train',
  weight_decay=1e-3,
  dropout_p=0.5,
  dropout_type='dropout',
  regularizer_type='l1',
  lr=0.0001,
  alpha=2,
  gamma=2,
  epochs=20,
  no_save=False,
  arch='vgg',
  save_path='experiments/attention_models',
  visualize=True,
  batch_size=32,
  type='status',
  num_workers=4,
  log_interval=100
)

run = wandb.init(settings=wandb.Settings(start_method="thread"), config=hyperparameter_defaults, project='kaokore-attention-sweeps', resume=True)
config = run.config

def train():

    name = 'kaokore-attention-sweeps'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print('Starting training')
    #with wandb.init(#settings=wandb.Settings(start_method="fork"),
    #                name=name,
    #                config=config) as run:
    #    config = run.config

    # Load data
    transform_train = transforms.Compose([

        # geometric transformation
        # transforms.RandomCrop(32, padding=4),

        # transforms.RandomHorizontalFlip(p=0.5),

        # kernel filters
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),

        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    print(config.dataset_path)

    # train_set = Kaokore('../../kaokore', 'train', label, transform=transform_train)
    train_set = ImageFolder(config.dataset_path, transform=transform_train)
    test_set = ImageFolder('../../kaokore_imagenet_style/'+config.type+'/dev', transform=transform_test)
    #test_set = Kaokore('../kaokore', 'dev', label, transform=transform_test)
    #viz_set = Kaokore('../../kaokore', 'test', label, transform=transform_test)
    viz_set = ImageFolder('../../kaokore_imagenet_style/' + config.type + '/test', transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=True, num_workers=config.num_workers)
    viz_loader = DataLoader(viz_set, batch_size=8, shuffle=True, num_workers=config.num_workers)

    # n_classes = len(train_set.cls_to_gen)
    # class_names = list(train_set.cls_to_gen.values())
    n_classes = len(train_set.classes)
    class_names = list(train_set.classes)
    print('Loaded data')
    print(class_names, n_classes)

    # Create model
    if config.arch == 'vgg':
        model = AttnVGG(num_classes=n_classes, output_layers=[0, 7, 21, 28], dropout_mode=config.dropout_type,
                        p=config.dropout_p).to(device)
    elif config.arch == 'resnet':
        model = AttnResnet(num_classes=n_classes, output_layers=['0', '4.1.4', '6.2.2', '7.1.2'],
                           dropout_mode=config.dropout_type, p=config.dropout_p).to(device)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    # Create loss criterion & optimize
    criterion = focal_loss(n_classes, config.gamma, config.alpha)  # 2, 2)
    # criterion = nn.CrossEntropyLoss()
    if config.regularizer_type == 'l2':
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # lr_lambda = lambda epoch : np.power(0.5, int(epoch/25))
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    for epoch in range(config.epochs):
        train_loss, train_acc, train_recall, train_prec, train_f1 = train_epoch(run, model, criterion, optimizer,
                                                                                train_loader, device, epoch,
                                                                                config.log_interval,
                                                                                config.regularizer_type,
                                                                                config.weight_decay)
        val_loss, val_acc, val_recall, val_prec, val_f1 = val_epoch(run, model, criterion, test_loader, device,
                                                                    epoch)
        # adjust learning rate
        # scheduler.step()

    else:
        if not config.no_save:
            torch.save(model.state_dict(),
                       os.path.join(config.save_path,
                                    name + '_' + config.arch + ".pth"))
            print("Saving Model to visualize")
    global_train_table = wandb.Table(
        columns=['experiment name', 'training loss', 'training accuracy', 'training recall', 'training precision',
                 'training f1'])
    global_train_table.add_data(name, train_loss, train_acc, train_recall, train_prec, train_f1)
    global_val_table = wandb.Table(
        columns=['experiment name', 'test loss', 'test accuracy', 'test recall', 'test precision', 'test f1'])
    global_val_table.add_data(name, val_loss, val_acc, val_recall, val_prec, val_f1)
    run.log({'Training end table': global_train_table, 'Validation end table': global_val_table})

    # visualize
    if config.visualize:
        model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(viz_loader):
                # get images
                inputs = inputs.to(device)
                if batch_idx == 0:
                    images = inputs[0:16, :, :, :]
                    I = utils.make_grid(images, nrow=4, normalize=True, scale_each=True)
                    _, c0, c1, c2, c3 = model(images)
                    print(I.shape, c0.shape, c1.shape, c2.shape, c3.shape)
                    attn0 = visualize_attn(I, c0)
                    attn1 = visualize_attn(I, c1)
                    attn2 = visualize_attn(I, c2)
                    attn3 = visualize_attn(I, c3)

                    viz_table = wandb.Table(
                        columns=['image', 'layer 0', 'low layer', 'middle layer', 'end layer'])

                    w_img = wandb.Image(I)
                    w_attn0 = wandb.Image(attn0)
                    w_attn1 = wandb.Image(attn1)
                    w_attn2 = wandb.Image(attn2)
                    w_attn3 = wandb.Image(attn3)

                    viz_table.add_data(w_img, w_attn0, w_attn1, w_attn2, w_attn3)
                    run.log({'visualization': viz_table})
                    break
            # confusion matrix
            print('Making the confusion matrix')
            cm = make_confusion_matrix(model, n_classes, test_loader, device)
            cm_img = plot_confusion_matrix(cm, class_names)
            w_cm = wandb.Image(cm_img)

            # log most and least confident images
            print('Logging the most and least confident images')
            (lc_scores, lc_imgs), (mc_scores, mc_imgs) = get_most_and_least_confident_predictions(model,
                                                                                                  test_loader,
                                                                                                  device)
            w_lc = wandb.Image(utils.make_grid(lc_imgs, nrow=4, normalize=True, scale_each=True))
            w_mc = wandb.Image(utils.make_grid(mc_imgs, nrow=4, normalize=True, scale_each=True))

            run.log({'Confusion Matrix': w_cm, 'Least Confident Images': w_lc, 'Most Confident Images': w_mc})
            print('Evaluations logged')
    run.finish()


if __name__ == '__main__':
    train()



