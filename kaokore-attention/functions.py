import torch
import torch.nn.functional as F
import torchvision.utils as utils
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import cv2

def train_epoch(run, model, criterion, optimizer, dataloader, device, epoch, log_interval, writer):
    model.train()
    losses = []
    all_label = []
    all_pred = []

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        # get the inputs and labels
        inputs, labels = inputs.to(device), labels.to(device) # bsz,3,h,w , bsz
        #print('dataset data', inputs.shape, labels.shape)

        optimizer.zero_grad()
        # forward
        outputs = model(inputs)# bsz,nclasses , bsz,1,h,w , bsz,1,h/2,w/2 , bsz,1,h/4,w/4
        #print('outputs', outputs[0].shape, outputs[1].shape, outputs[2].shape, outputs[3].shape, len(outputs))
        if isinstance(outputs, list):
            outputs = outputs[0]

        # compute the loss
        loss = criterion(outputs, labels.squeeze())
        losses.append(loss.item())

        # compute the accuracy
        prediction = torch.max(outputs, 1)[1]# bsz
        all_label.extend(labels.squeeze())
        all_pred.extend(prediction)
        score = accuracy_score(labels.squeeze().cpu().data.squeeze().numpy(),
                                   prediction.cpu().data.squeeze().numpy())


        # backward & optimize
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            print("epoch {:3d} | iteration {:5d} | Loss {:.6f} | Acc {:.2f}%".format(epoch+1, batch_idx+1, loss.item(), score*100))

    # Compute the average loss & accuracy
    training_loss = sum(losses)/len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    training_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    training_recall = recall_score(all_label.squeeze().cpu().data.squeeze().numpy(),
                                  all_pred.cpu().data.squeeze().numpy())
    training_precision = precision_score(all_label.squeeze().cpu().data.squeeze().numpy(),
                                  all_pred.cpu().data.squeeze().numpy())
    training_f1 = f1_score(all_label.squeeze().cpu().data.squeeze().numpy(),
                                  all_pred.cpu().data.squeeze().numpy())
    # Log
    run.log({'Train Loss': training_loss, 'Train Accuracy': training_acc})
    print("Average Training Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch+1, training_loss, training_acc*100))

    return training_loss, training_acc, training_recall, training_precision, training_f1


def val_epoch(run, model, criterion, dataloader, device, epoch, writer):
    model.eval()
    losses = []
    all_label = []
    all_pred = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # get the inputs and labels
            inputs, labels = inputs.to(device), labels.to(device)
            # forward
            outputs = model(inputs)
            if isinstance(outputs, list):
                outputs = outputs[0]
            # compute the loss
            loss = criterion(outputs, labels.squeeze())
            losses.append(loss.item())
            # collect labels & prediction
            prediction = torch.max(outputs, 1)[1]
            all_label.extend(labels.squeeze())
            all_pred.extend(prediction)

    # Compute the average loss & accuracy
    val_loss = sum(losses)/len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    val_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    val_recall = recall_score(all_label.squeeze().cpu().data.squeeze().numpy(),
                                   all_pred.cpu().data.squeeze().numpy())
    val_precision = precision_score(all_label.squeeze().cpu().data.squeeze().numpy(),
                                         all_pred.cpu().data.squeeze().numpy())
    val_f1 = f1_score(all_label.squeeze().cpu().data.squeeze().numpy(),
                           all_pred.cpu().data.squeeze().numpy())
    # Log
    writer.add_scalars('Loss', {'val': val_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'val': val_acc}, epoch+1)
    run.log({'Validation Loss': val_loss, 'Validation Accuracy':val_acc})
    print("Average Validation Loss: {:.6f} | Acc: {:.2f}%".format(val_loss, val_acc*100))

    return val_loss, val_acc, val_recall, val_precision, val_f1


def visualize_attn(I, c):
    # Image
    img = I.permute((1,2,0)).cpu().numpy()
    # Heatmap
    N, C, H, W = c.size()
    a = F.softmax(c.view(N,C,-1), dim=2).view(N,C,H,W)
    up_factor = 32/H
    # print(up_factor, I.size(), c.size())
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=4, normalize=True, scale_each=True)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    # Add the heatmap to the image
    vis = 0.6 * img + 0.4 * attn
    return torch.from_numpy(vis).permute(2,0,1)
