import torch
import numpy as np
from sklearn.metrics import classification_report
import itertools
import matplotlib.pyplot as plt
from torchvision.io import read_image


def lr_scheduler(epoch, args):
    lr = args.lr * (0.1**(epoch // args.lr_adjust_freq))
    return lr


# evaluating the model
def evaluate(data_loader, model):
    with torch.no_grad():
        progress = ["/", "-", "\\", "|", "/", "-", "\\", "|"]
        model.eval().cuda()
        true_y, pred_y = [], []
        for i, batch_ in enumerate(data_loader):
            X, y = batch_
            print(progress[i % len(progress)], end="\r")
            y_pred = (model(X.cuda()).sigmoid() > 0.5).int().squeeze()#threshold of 0.5 chosen
            #print(y.shape, y_pred.shape)
            true_y.extend(y.squeeze().cpu())
            pred_y.extend(y_pred.squeeze().cpu())
        true_y = torch.stack(true_y)
        pred_y = torch.stack(pred_y)
        print(true_y.shape, pred_y.shape)
        return classification_report(true_y, pred_y, digits=3, output_dict=True)


def make_confusion_matrix(module, loader, device):
    @torch.no_grad()
    def get_all_preds(model):
        all_preds = torch.tensor([]).to(device)
        all_tgts = torch.tensor([]).to(device)
        for batch in loader:
            images, labels = batch
            preds = model(images.to(device))
            all_preds = torch.cat(
                (all_preds, preds)
                , dim=0
            )
            all_tgts = torch.cat(
                (all_tgts, labels.to(device))
                , dim=0
            )
        return all_preds, all_tgts

    # set up model predictions and targets in right format for making the confusion matrix
    preds, tgts = get_all_preds(module.model.to(device))
    stacked = torch.stack(
        (
            tgts.squeeze()
            , (preds.sigmoid() > 0.5).int().squeeze()  # threshold of 0.5 chosen
        )
        , dim=1
    )

    # make the confusion matrix
    cm = torch.zeros(module.hparam.n_classes, module.hparam.n_classes, dtype=torch.int64)
    for p in stacked:
        tl, pl = p.tolist()
        cm[int(tl), int(pl)] = cm[int(tl), int(pl)] + 1

    return cm


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        #print('Confusion matrix, without normalization')
        pass

    # set up the confusion matrix visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('temp_cm_logging.jpg')
    plt.close('all')
    return read_image('temp_cm_logging.jpg')/255


def get_most_and_least_confident_predictions(model, loader, device):
    # get model prediction for the validation dataset and all images for later use
    preds = torch.tensor([]).to(device)
    tgts = torch.tensor([]).to(device)
    all_images = torch.tensor([]).to(device)
    #loader = module.val_dataloader()
    #model = module.model.to(device)
    for batch in loader:
        images, y = batch
        pred_batch = model(images.to(device))
        preds = torch.cat(
            (preds, pred_batch)
            , dim=0
        )
        tgts = torch.cat(
            (tgts, y.to(device))
            , dim=0
        )
        all_images = torch.cat(
            (all_images, images.to(device)),
            dim=0
        )
    #print(preds.shape, tgts.shape)
    confidence = (preds.sigmoid()-tgts).abs()
    print(confidence.shape)

    # get indices with most and least confident scores
    mc_scores, most_confident = confidence.topk(4, dim=0)
    lc_scores, least_confident = confidence.topk(4, dim=0, largest=False)

    # get the images according to confidence scores, 4 each
    mc_imgs = all_images[most_confident.squeeze()]
    lc_imgs = all_images[least_confident.squeeze()]

    return (mc_scores, mc_imgs), (lc_scores, lc_imgs)