import argparse
from torch.optim import Adam, SGD
from models import *
from utils import *
from dataset import Kaokore, gen_val_transforms, gen_train_transforms

import warnings
from argparse import Namespace
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback

from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from torch.utils.data import DataLoader, Subset
from torch.multiprocessing import cpu_count
from torch.optim import RMSprop
from torch.optim.lr_scheduler import LambdaLR

from torchvision import transforms
from torchvision.datasets import ImageFolder

from models import FinetunedModel
from utils import *
#from ds_augmentations import AugDatasetWrapper
import pandas as pd


# training loop components
class FinetunedClassifierModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.hparam = hparams
        self.model = setup_base_models(hparams.arch, hparams.n_classes)
        self.loss = nn.CrossEntropyLoss()
        #self.device = device


    def total_steps(self):
        return len(self.train_dataloader()) // self.hparam.epochs

    def get_transforms(self, split):
        return gen_val_transforms(self.hparam)
        #if split=='train':
        #    return gen_train_transforms(self.hparam)
        #else:
        #    return gen_val_transforms(self.hparam)


    def get_dataloader(self, split):

        split_ds = Kaokore(self.hparam, split, self.hparam.label, transform=self.get_transforms(split))

        print(split_ds[-1], self.hparam.batch_size, split, cpu_count())
        #temp = DataLoader(split_ds, batch_size=2, num_workers= 1, shuffle=True)
        #print('loaded successfully')
        return DataLoader(
            split_ds,
            batch_size=self.hparam.batch_size,
            shuffle=split == "train",
            num_workers=cpu_count()
            drop_last=False)

    def train_dataloader(self):
        return self.get_dataloader("train")

    def val_dataloader(self):
        return self.get_dataloader("dev")

    def test_dataloader(self):
        return self.get_dataloader("test")

    def forward(self, x):
        return self.model(x)

    def step(self, batch, step_name="train"):
        x, y = batch
        y_out = self.forward(x)
        loss = self.loss(y_out, y)
        loss_key = f"{step_name}_loss"
        tensorboard_logs = {loss_key: loss}
        self.log('training_metrics', tensorboard_logs)  # this is needed to get the logs

        return {("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
                "progress_bar": {loss_key: loss}}

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "dev")

    def validation_end(self, outputs):
        
        if len(outputs) == 0:
            return {"val_loss": torch.tensor(0)}
        else:
            loss = torch.stack([x["val_loss"] for x in outputs]).mean()
            return {"val_loss": loss, "log": {"val_loss": loss}}

    def choose_optimizer(self, params):
        choices = ['sgd', 'adam']
        if self.hparam.optimizer=='sgd':
            return SGD(params, self.hparam.lr, self.hparam.momentum, weight_decay=self.hparam.wd)
        elif self.hparam.optimizer=='adam':
            return Adam(params, self.hparam.lr, weight_decay=self.hparam.wd)

    
    def configure_optimizers(self):
        optimizer = self.choose_optimizer(self.model.parameters())
        lambda_sched = lambda epoch: lr_scheduler(epoch, self.hparam)
        schedulers = [
            #CosineAnnealingLR(optimizer, self.hparam.epochs),
            LambdaLR(optimizer, lambda_sched)
        ] if self.hparam.epochs > 1 else []
        return [optimizer], schedulers


class LogPredictionsCallback(Callback):

    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx):

        #confusion matrix
        cm = make_confusion_matrix(module, module.val_dataloader(), module.device)
        cm_img = plot_confusion_matrix(cm, module.hparam.class_names)
        trainer.logger.log_image(key='Confusion matrix', images = [cm_img], caption=['Confusion matrix for shakespeare and queen victoria images'])

        # log most and least confident images
        (lc_scores, lc_imgs), (mc_scores, mc_imgs) = get_most_and_least_confident_predictions(module.model, module.val_dataloader(), module.device)
        lc_captions = [f'Confidence score: {score}' for score in lc_scores]
        trainer.logger.log_image(key='Least Confident Images', images = [img for img in lc_imgs], caption=lc_captions)
        mc_captions = [f'Confidence score: {score}' for score in mc_scores]
        trainer.logger.log_image(key='Most Confident Images', images=[img for img in mc_imgs], caption=mc_captions)
    
    
    def on_train_end(self, trainer, module):
        eval_metrics = evaluate(module.val_dataloader(), module.model)
        classes = module.hparam.class_names
        data=[]
        tkeys = list(eval_metrics.keys())
        print(tkeys, len(classes))
        print(eval_metrics)

        labels = ['Category Name']+list(eval_metrics[tkeys[0]].keys())
        for i in range(len(classes)):
            data[i].append(eval_metrics[tkeys[i]].values())

        #df = pd.DataFrame(eval_metrics).transpose()
        
        
        trainer.logger.log_table(key="Evaluation metrics", columns=list(labels), data=data)
        
        module.log('total_accuracy', eval_metrics['accurcy'])
        #trainer.logger.experiment.summary(eval_metrics)
        #trainer.logger.experiment.log(eval_metrics)


def train(args, device):
    #train_idx, val_idx, n_classes,classes_names = get_train_val_split(args, get_n_classes = True)
    gen_to_cls = {'male': 0, 'female': 1} if args.label == 'gender' else {'noble': 0, 'warrior': 1, 'incarnation': 2, 'commoner': 3}

    # using the suggested lr
    hparams_cls = Namespace(
        arch=args.arch,

        lr=args.lr,
        momentum=args.momentum,
        wd=args.wd,
        optimizer=args.optimizer,
        lr_adjust_freq=args.lr_adjust_freq,

        epochs=args.epochs,

        label=args.label,
        batch_size=args.batch_size,
        root=args.root,
        image_size=(args.image_size, args.image_size),
        n_classes=len(gen_to_cls),
        class_names=gen_to_cls.keys(),

        hidden_size=args.hidden_size,
        freeze_base=args.freeze_base,
        
        device = device,
    )

    df = pd.read_csv(f'{args.root}/labels.csv')
    image_dir = f'{args.root}/images_256/'

    module = FinetunedClassifierModule(hparams_cls)

    logger = WandbLogger(project=args.experiment_name,
                         name=args.ver_name,
                         config={
                             'learning_rate': args.lr,
                             'architecture': args.arch,
                             'dataset': 'Kaokore',
                             'epochs': args.epochs,
                         })
    logger.watch(module, log='all', log_freq=args.log_interval)

    trainer = pl.Trainer(gpus=1,
                         max_epochs=hparams_cls.epochs,
                         logger=logger,
                         callbacks= [LogPredictionsCallback()],
                         log_every_n_steps=args.log_interval)  # need the last arg to log the training iterations per step

    print(next(iter(module.train_dataloader())))
    print('Training the model')
    trainer.fit(module)

    print('finished training')
    
    
if __name__=='__main__':

    #modify this for possible architectures:
    model_choices= ['vgg', 'resnet', 'mobilenet', 'densenet']


    parser = argparse.ArgumentParser(description="Train a Keras model on the KaoKore dataset")
    parser.add_argument('--arch', type=str, choices=model_choices, required=True)
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='enables/disables CUDA training')
    
    parser.add_argument('--label', type=str, choices=['gender', 'status'], required=True)
    parser.add_argument('--root', type=str, required=True)

    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='Number of workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--lr-adjust-freq' , type=int, default=10, help='How many epochs per LR adjustment (*=0.1)')
    parser.add_argument('--hidden-size', type=int, default=512, metavar='N',
                        help='hidden size of the fc layer of the model (default: 512)')
    parser.add_argument('--freeze-base', action='store_true', default=True,
                        help='Freeze the pretrained model before training? (default: True)')

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (L2 penalty)')

    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status (default: 50)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--experiment-name', type=str, default = 'finetuning-classifier-on-kaokore',
                        help='Model experiment name')
    parser.add_argument('--ver-name', type=str, default = 'proto_v0',
                        help="Model experiment's version/run name")

    args = parser.parse_args()

    num_classes = 2 if args.label == 'gender' else 4

    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    train(args, device)

    