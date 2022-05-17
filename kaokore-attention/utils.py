import torch
import torch.nn.functional as F
from torchvision.io import read_image
import matplotlib.pyplot as plt
import itertools
import numpy as np

#drops images in a batch from being seen by a subset of layers
def drop_connect(inputs, p, train_mode):  # bchw, 0-1, bool
    assert 0 <= p <= 1, 'p must be in range of [0,1]'

    #inference
    if not train_mode:
        return inputs

    bsz = inputs.shape[0]

    keep_prob = 1-p

    #binary mask for selection of weights
    rand_tensor = keep_prob
    dims = len(inputs.shape)-1
    rand_tensor += torch.rand([bsz,]+[1,]*dims, dtype=inputs.dtype, device=inputs.device)
    mask = torch.floor(rand_tensor)

    outputs = inputs / keep_prob*mask
    return outputs

def focal_loss(n_classes, gamma=2., alpha=4.):
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y, y_pred):
        eps = 1e-9
        pred = torch.softmax(y_pred, dim=1) + eps
        #pred = y_pred + eps
        y_true = F.one_hot(y, n_classes)
        cross_entropy = y_true * -1*torch.log(pred)
        wt = y_true*(1-pred)**gamma
        focal_loss = alpha*wt*cross_entropy
        focal_loss = torch.max(focal_loss, dim=1)[0]
        return focal_loss.mean()
    return focal_loss_fixed


def make_confusion_matrix(model, n_classes, loader, device):
    @torch.no_grad()
    def get_all_preds(model):
        all_preds = torch.tensor([]).to(device)
        all_tgts = torch.tensor([]).to(device)
        for batch in loader:
            images, labels = batch
            preds = model(images.to(device))[0]
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
    preds, tgts = get_all_preds(model.to(device))
    #print(preds.argmax(dim=1).shape, tgts.shape)
    stacked = torch.stack(
        (
            tgts.squeeze()
            , preds.argmax(dim=1)
        )
        , dim=1
    )

    # make the confusion matrix
    cm = torch.zeros(n_classes, n_classes, dtype=torch.int64)
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
    plt.savefig('misc/temp_cm_logging.jpg')
    plt.close('all')
    return read_image('misc/temp_cm_logging.jpg')/255

def get_most_and_least_confident_predictions(model, loader, device):
    # get model prediction for the validation dataset and all images for later use
    preds = torch.tensor([]).to(device)
    tgts = torch.tensor([]).to(device)
    all_images = torch.tensor([]).to(device)
    #loader = module.val_dataloader()
    #model = module.model.to(device)
    for batch in loader:
        images, y = batch
        pred_batch = model(images.to(device))[0]
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
    confidence = F.softmax(preds, dim=1).max(dim=1)[0]
    #print(confidence.shape)

    # get indices with most and least confident scores
    lc_scores, least_confident = confidence.topk(4, dim=0)
    mc_scores, most_confident = confidence.topk(4, dim=0, largest=False)

    # get the images according to confidence scores, 4 each
    mc_imgs = all_images[most_confident.squeeze()]
    lc_imgs = all_images[least_confident.squeeze()]

    return (mc_scores, mc_imgs), (lc_scores, lc_imgs)