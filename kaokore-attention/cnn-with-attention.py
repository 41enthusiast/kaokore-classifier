import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils
from tensorboardX import SummaryWriter
import os
import argparse
import numpy as np
from datetime import datetime
from models import AttnVGG
from functions import train_epoch, val_epoch, visualize_attn

from dataset import Kaokore

model_choices = ['vgg', 'resnet', 'densenet']

# Parameters manager
parser = argparse.ArgumentParser(description='CNN with Attention')
parser.add_argument('--train', action='store_true',
    help='Train the network')
parser.add_argument('--visualize', action='store_true',
    help='Visualize the attention vector')
parser.add_argument('--no_save', action='store_true',
    help='Not save the model')
parser.add_argument('--save_path', default='experiments/attention_models', type=str,
    help='Path to save the model')
parser.add_argument('--checkpoint', default='cnn_checkpoint.pth', type=str,
    help='Path to checkpoint')
parser.add_argument('--arch', type=str, choices=model_choices, required=True)
parser.add_argument('--epochs', default=300, type=int,
    help='Epochs for training')
parser.add_argument('--batch_size', default=32, type=int,
    help='Batch size for training or testing')
parser.add_argument('--lr', default=1e-4, type=float,
    help='Learning rate for training')
parser.add_argument('--weight_decay', default=1e-4, type=float,
    help='Weight decay for training')
#parser.add_argument('--device', default='0', type=str,
#    help='Cuda device to use')
parser.add_argument('--log_interval', default=100, type=int,
    help='Interval to print messages')
args = parser.parse_args()

# Use specific gpus
#os.environ["CUDA_VISIBLE_DEVICES"]=args.device
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    label = 'status'
    experiment_name='_'.join(['noimgcrop',args.arch, label, str(args.lr), str(args.epochs)])
    run = wandb.init(project='kaokore-model-attention', name=experiment_name,
               config={'type': label,
                       #'image-size': args.image_size,
                       'max-epochs': args.epochs,
                       'lr': args.lr,
                       'batch-size': args.batch_size})

    # Load data
    transform_train = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    transform_test = transforms.Compose([
        #transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    train_set = Kaokore('../../kaokore-classifier/kaokore', 'train', label, transform=transform_train)
    test_set = Kaokore('../../kaokore-classifier/kaokore', 'test', label, transform=transform_test)
    #train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=16)
    #test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=16)
    # Create model
    model = AttnVGG(num_classes=len(train_set.cls_to_gen), output_layers=[0, 7, 21, 28]).to(device)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    # Summary writer
    writer = SummaryWriter("runs/cnn_attention_{:%Y-%m-%d_%H-%M-%S}".format(datetime.now()))
    # Train
    if args.train:
        # Create loss criterion & optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # lr_lambda = lambda epoch : np.power(0.5, int(epoch/25))
        # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        for epoch in range(args.epochs):
            train_loss, train_acc, train_recall, train_prec, train_f1 = train_epoch(run, model, criterion, optimizer, train_loader, device, epoch, args.log_interval, writer)
            val_loss, val_acc, val_recall, val_prec, val_f1 = val_epoch(run, model, criterion, test_loader, device, epoch, writer)
            # adjust learning rate
            # scheduler.step()
            if not args.no_save:
                torch.save(model.state_dict(), os.path.join(args.save_path, "cnn_epoch{:03d}.pth".format(epoch+1)))
                print("Saving Model of Epoch {}".format(epoch+1))
        global_train_table = wandb.Table(columns=['experiment name', 'training loss', 'training accuracy', 'training recall', 'training precision', 'training f1'])
        global_train_table.add_data(experiment_name, train_loss, train_acc, train_recall, train_prec, train_f1)
        global_val_table = wandb.Table(
            columns=['experiment name', 'test loss', 'test accuracy', 'test recall', 'test precision', 'test f1'])
        global_val_table.add_data(experiment_name, val_loss, val_acc, val_recall, val_prec, val_f1)
        run.log({'Training end table': global_train_table, 'Validation end table': global_val_table})

    # Visualize
    if args.visualize:
        # Load model
        model.load_state_dict(torch.load(args.checkpoint))
        model.eval()

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                # get images
                inputs = inputs.to(device)
                if batch_idx == 0:
                    images = inputs[0:16,:,:,:]
                    I = utils.make_grid(images, nrow=4, normalize=True, scale_each=True)
                    #writer.add_image('origin', I)
                    _, c0, c1, c2, c3 = model(images)
                    print(I.shape, c0.shape, c1.shape, c2.shape, c3.shape)
                    attn0 = visualize_attn(I, c0)
                    attn1 = visualize_attn(I, c1)
                    #writer.add_image('attn1', attn1)
                    attn2 = visualize_attn(I, c2)
                    #writer.add_image('attn2', attn2)
                    attn3 = visualize_attn(I, c3)
                    #writer.add_image('attn3', attn3)

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
