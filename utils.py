import torch
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import matthews_corrcoef
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import math

def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def write_confusion_matrix(writer:SummaryWriter, true_values:np.ndarray, 
                            pred_values:np.ndarray, 
                            e: int, message:str):
    ConfusionMatrixDisplay.from_predictions(true_values, pred_values)
    mcc = matthews_corrcoef(true_values,pred_values)
    plt.title('MCC= '+str(mcc))
    writer.add_figure(message, plt.gcf(),e,True)

def find_lr(model, train_loader, loss_fn, optimizer, init_value=1e-8, final_value=10.0,
             device = 'cuda'):
    number_in_epoch = len(train_loader) - 1
    update_step = (final_value / init_value) ** (1 / number_in_epoch)
    lr = init_value
    optimizer.param_groups[0]["lr"] = lr
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    for X,y in train_loader:
        batch_num += 1
        X, y = X.to(device).float(), y.to(device).float()
        #inputs, labels = inputs, labels
        optimizer.zero_grad()
        outputs = torch.squeeze(model(torch.unsqueeze(X,1).float()))
        loss = loss_fn(outputs, y).float()

        # Crash out if loss explodes

        if batch_num > 1 and loss > 4 * best_loss:
            return log_lrs[10:-5], losses[10:-5]

        # Record the best loss

        if loss < best_loss or batch_num == 1:
            best_loss = loss

        # Store the values

        losses.append(loss)
        log_lrs.append(math.log10(lr))

        # Do the backward pass and optimize

        loss.backward()
        optimizer.step()

        # Update the lr for the next step and store

        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr
    return log_lrs[10:-5], losses[10:-5]