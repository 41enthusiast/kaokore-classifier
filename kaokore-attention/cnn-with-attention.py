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

model_choices = ['vgg', 'resnet', 'densenet']
regularizer_choices = ['l1', 'l2']
dropout_choices = ['dropconnect', 'dropout']

# Parameters manager
parser = argparse.ArgumentParser(description='CNN with Attention')
parser.add_argument('--experiment-name', default='ka', type=str,
                    help='Name of the experiment with different configurations')
parser.add_argument('--train', action='store_true',
                    help='Train the network')
parser.add_argument('--visualize', action='store_true',
                    help='Visualize the attention vector')
parser.add_argument('--no_save', action='store_true',
                    help='Not save the model')
parser.add_argument('--save_path', default='experiments/attention_models', type=str,
                    help='Path to save the model')
# parser.add_argument('--checkpoint', default='cnn_checkpoint.pth', type=str,
#    help='Path to checkpoint')
parser.add_argument('--arch', type=str, choices=model_choices, required=True,
                    help='Name of base architecture/feature extractor. choices: vgg, resnet')
# parser.add_argument('--regularizer-type', type=str, choices=regularizer_choices, required=True,
#                    help='Type of regularization (l1/l2)')
# parser.add_argument('--dropout-type', type=str, choices=dropout_choices, required=True,
#                    help='Type of dropout to use (dropout, dropconnect)')
# parser.add_argument('--dropout-p', default=0.2, type=float,
#    help='Dropout probability value (default: 0.2)')
parser.add_argument('--epochs', default=300, type=int,
                    help='Epochs for training')
parser.add_argument('--num-workers', default=16, type=int,
                    help='Number of workers (default: 16)')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training or testing')
# parser.add_argument('--lr', default=1e-4, type=float,
#    help='Learning rate for training')
# parser.add_argument('--weight-decay', default=1e-4, type=float,
#    help='Weight decay for training')
# parser.add_argument('--device', default='0', type=str,
#    help='Cuda device to use')
parser.add_argument('--log_interval', default=100, type=int,
                    help='Interval to print messages')
args = parser.parse_args()

# Use specific gpus
# os.environ["CUDA_VISIBLE_DEVICES"]=args.device
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(config=None):

    name = args.experiment_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print('Starting training')
    with wandb.init(settings=wandb.Settings(start_method="fork"),
                    name=name,
                    config=config) as run:
        config = run.config

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
        test_set = ImageFolder('../kaokore_imagenet_style/'+label+'/dev', transform=transform_test)
        #test_set = Kaokore('../kaokore', 'dev', label, transform=transform_test)
        #viz_set = Kaokore('../../kaokore', 'test', label, transform=transform_test)
        viz_set = ImageFolder('../kaokore_imagenet_style/' + label + '/test', transform=transform_test)
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

        for epoch in range(args.epochs):
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


if __name__ == '__main__':
    label = 'status'

    sweep_config = {
        'method': 'grid',
    }

    parameters_dict = {
        'dataset_path': {
            'values': ['../../kaokore_imagenet_style/'+label+'/train',
                        '../../fst-kaokore-1-ub',
                       '../../fst-kaokore-2-ub',
                       '../../fst-kaokore-cb',
                       '../../fst-kaokore-2-cb']
        },

        'weight_decay': {
            'values': [1e-3]  # [10, 1e-1, 1e-3, 1e-4]
        },

        'dropout_p': {
            'values': [0, 0.3, 0.5]  # [0.5, 0.3, 0.1, 0.0]
        },

        'dropout_type': {
            'values': dropout_choices
        },

        'regularizer_type': {
            'values': regularizer_choices
        },

        'lr': {
            'values': [0.0001, 0.01]  # [0.0001, 0.001, 0.01, 0.1]
        },

        'alpha': {
            'values': [1, 2, 5]
        },

        'gamma': {
            'values': [5, 2, 0]  # [5, 2, 1, 0]
        },

        'epochs': {
            'value': args.epochs
        },
        'no_save': {
            'value': args.no_save
        },
        'arch': {
            "value": args.arch
        },
        'save_path': {
            "value": args.save_path
        },
        'visualize': {
            'value': args.visualize
        },
        'batch_size': {
            'value': args.batch_size
        },
        'type': {
            'value': label
        },
        'num_workers': {
            'value': args.num_workers
        },
        'log_interval': {
            'value': args.log_interval
        }

    }
    sweep_config['parameters'] = parameters_dict

    sweep_config['metric'] = {
        'name': 'Validation Loss',
        'goal': 'minimize'
    }

    sweep_config['early_terminate'] = {
        'type': 'hyperband',
        'min_iter': 3,
        #'max_iter': args.epochs,
        #'s': 6,
        #'eta': 1
    }

    # experiment_name='_'.join([args.experiment_name, args.arch, label, str(args.lr), str(args.epochs), args.regularizer_type, args.dropout_type, str(args.dropout_p)])
    # run = wandb.init(project='kaokore-model-attention', name=experiment_name,
    #           config=exp_config)

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

    transform_viz = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # train_set = Kaokore('../../kaokore', 'train', label, transform=transform_train)
    # train_set = ImageFolder('../../kaokore-style-transfer-aug-ds', transform=transform_train)
    # test_set = Kaokore('../../kaokore', 'test', label, transform=transform_test)
    # viz_set = Kaokore('../../kaokore', 'test', label, transform=transform_viz)
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
    #num_workers = args.num_workers)
    # test_loader = DataLoader(test_set, batch_size=8, shuffle=True, num_workers=args.num_workers)
    # viz_loader = DataLoader(viz_set, batch_size=8, shuffle=True, num_workers=args.num_workers)

    # n_classes = len(train_set.cls_to_gen)
    # class_names = list(train_set.cls_to_gen.values())
    # n_classes = len(train_set.classes)
    # class_names = list(train_set.classes)

    # parameters_dict.update({
    #    'n_classes': {
    #        'value': n_classes
    #    },
    #    'class_names': {
    #        'value': class_names
    #    }
    # })
    sweep_id = wandb.sweep(sweep_config, project='kaokore-model-attention')

    # Train
    if args.train:
        wandb.agent(sweep_id, function=train, count=4)



