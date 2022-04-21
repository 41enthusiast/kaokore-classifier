import torch
import numpy as np
from sklearn.metrics import classification_report
import itertools
import matplotlib.pyplot as plt
from torchvision.io import read_image
import torch.nn.functional as F


def lr_scheduler(epoch, args):
    lr = args.lr * (0.1**(epoch // args.lr_adjust_freq))
    return lr


# evaluating the model
def evaluate(data_loader, model, n_classes):
    with torch.no_grad():
        progress = ["/", "-", "\\", "|", "/", "-", "\\", "|"]
        model.eval().cuda()
        true_y, pred_y = [], []
        for i, batch_ in enumerate(data_loader):
            x, y = batch_
            print(progress[i % len(progress)], end="\r")
            y_pred = model(x.cuda()).argmax(dim=1)
            #print(y.shape, y_pred.shape)
            true_y.extend(y.squeeze().cpu())
            pred_y.extend(y_pred.cpu())
        true_y = torch.stack(true_y)
        pred_y = torch.stack(pred_y)

        print(true_y.shape, pred_y.shape)

        stacked = torch.stack((true_y,  pred_y), dim=1)

        # make the confusion matrix
        cm = torch.zeros(n_classes, n_classes, dtype=torch.int64)
        for p in stacked:
            tl, pl = p.tolist()
            cm[int(tl), int(pl)] = cm[int(tl), int(pl)] + 1

        print()
        class_precision = lambda mat, index: mat[index, index]/(mat[:, index].sum())
        class_recall = lambda mat, index: mat[index, index] / (mat[index, :].sum())

        per_class_acc = cm.diag() / cm.sum(1)
        per_class_precision = [class_precision(cm, i) for i in range(len(cm))]
        per_class_recall = [class_recall(cm, i) for i in range(len(cm))]

        total_acc = true_y == pred_y
        total_acc = sum(total_acc)/len(total_acc)
        global_stats = {'total accuracy': total_acc}

        print(per_class_acc, per_class_precision, per_class_recall, global_stats)

        return per_class_acc, per_class_precision, per_class_recall, global_stats


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
    #print(preds.argmax(dim=1).shape, tgts.shape)
    stacked = torch.stack(
        (
            tgts.squeeze()
            , preds.argmax(dim=1)
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
    confidence = F.softmax(preds, dim=1).max(dim=1)[0]
    #print(confidence.shape)

    # get indices with most and least confident scores
    lc_scores, least_confident = confidence.topk(4, dim=0)
    mc_scores, most_confident = confidence.topk(4, dim=0, largest=False)

    # get the images according to confidence scores, 4 each
    mc_imgs = all_images[most_confident.squeeze()]
    lc_imgs = all_images[least_confident.squeeze()]

    return (mc_scores, mc_imgs), (lc_scores, lc_imgs)





def layer_grad_contrib(optimizer, model, loader, criterion, device):
    for param in model.parameters():
        param.requires_grad = True
    avgs_grad_of_params = {}
    for batch in loader:
        optimizer.zero_grad()
        x, y = batch
        pred = model(x.to(device))
        loss = criterion(pred, y.to(device))
        loss.backward()
        for name, parameter in model.named_parameters():
            #print(name)
            #print(parameter)
            #print(parameter.grad)
            if name in avgs_grad_of_params:
                avgs_grad_of_params[name] += [parameter.grad.abs().mean()]
            else:
                avgs_grad_of_params[name] = [parameter.grad.abs().mean()]
        avg_grad_of_params = {}
        ttl = 0.0
        for i in avgs_grad_of_params:
            avg_grad_of_params[i]=torch.stack(avgs_grad_of_params[i]).mean()
            ttl += avg_grad_of_params[i]
        rel_importance = {i: (avg_grad_of_params[i]/ttl*100).cpu().numpy() for i in avg_grad_of_params}
    model.eval()

    #make the bar plot
    plt.barh(y=[i for i in range(len(rel_importance))],
             width=list(rel_importance.values()),
             )
    plt.ylabel('Relative Layerwise contribution to loss')
    plt.xlabel('Percentage')
    plt.yticks([i for i in range(len(rel_importance))],
               list(rel_importance.keys()),
               rotation='horizontal'
               )
    plt.tight_layout()
    plt.savefig('misc/lyr_contrib_logging.jpg')
    plt.close('all')
    return read_image('misc/lyr_contrib_logging.jpg') / 255