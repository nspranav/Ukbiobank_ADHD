#%%
import torch
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import matthews_corrcoef,f1_score,balanced_accuracy_score
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import math
import plotly.graph_objects as go
#%%
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
    f1_s = f1_score(true_values,pred_values)
    b_a = balanced_accuracy_score(true_values, pred_values)
    tn, fp, fn, tp = confusion_matrix(true_values, pred_values).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp/ (tp+fn)

    plt.title('MCC= '+f'{mcc:.3f} F1= {f1_s:.3f} BAC= {b_a:.5f}')
    writer.add_figure(message, plt.gcf(),e,True)
    return mcc,f1_s,b_a,specificity,sensitivity

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

def plot_errorBar() -> None:
    """
    """
    x = [1,2,3,4]
    acc_1 = [3.014,3.139,3.087,3.053,2.971]
    val_1 = [2.960,3.025, 3.093,3.038,3.177]
    acc_2 = [4.441,4.413,4.600,4.410,4.475]
    val_2 = [4.677,4.495,4.379,4.629,4.206]
    acc_3 = [6.214,6.264,6.133,6.336,6.314]
    val_3 = [6.060,6.753,6.849,6.604,6.570]
    acc_4 = [8.332,8.197,8.155,8.061,8.187]
    val_4 = [9.179,8.256,8.886,8.638,8.179]
    y_acc = [np.mean(acc_1),np.mean(acc_2),np.mean(acc_3),np.mean(acc_4)]
    y_acc_err = [np.std(acc_1),np.std(acc_2),np.std(acc_3),np.std(acc_4)]
    y_val = [np.mean(val_1),np.mean(val_2),np.mean(val_3),np.mean(val_4)]
    y_val_err = [np.std(val_1),np.std(val_2),np.std(val_3),np.std(val_4)]

    fig = go.Figure(data=[go.Scatter(
        name = 'Training',
        x=[1, 2, 3, 4],
        y=y_acc,
        error_y=dict(
            type='data', # value of error bar given in data coordinates
            array=y_acc_err,
            visible=True)
    ),
    go.Scatter(
        name = 'Validation',
        x=[1,2,3,4],
        y=y_val,
        error_y=dict(
            type='data', # value of error bar given in data coordinates
            array=y_val_err,
            visible=True)
    )],
    layout= dict(
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = 1
        
    ),
    yaxis = dict(
        
    ),
    xaxis_title = 'No. of fully-connected layers',
    yaxis_title = 'L1 Loss',
    legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
    ) 
    )
    )
    fig.show() 

def plot_parameter_WM():
    x = [1,2,3,4,5]

    y_acc = [0.72, 0.6967, 0.6838, 0.6446, 0.5875]
    y_acc_err = [0.029, 0.0136, 0.0095, 0.0060, 0.0456]
    y_val = [0.663, 0.661, 0.6614, 0.6468, 0.6155]
    y_val_err = [0.007, 0.0065, 0.0104, 0.0097, 0.0423]
    y_test = [0.674, 0.6688, 0.6498, 0.6185, 0.6004 ]
    y_test_err = [0.0331, 0.017, 0.0236, 0.0131, 0.0616]

    fig = go.Figure(data=[go.Scatter(
        name = 'Training',
        x=[1, 2, 3, 4, 5],
        y=y_acc,
        error_y=dict(
            type='data', # value of error bar given in data coordinates
            array=y_acc_err,
            visible=True)
    ),
    go.Scatter(
        name = 'Validation',
        x=[1,2,3,4,5],
        y=y_val,
        error_y=dict(
            type='data', # value of error bar given in data coordinates
            array=y_val_err,
            visible=True)
    ),
     go.Scatter(
        name = 'Test',
        x=[1,2,3,4,5],
        y=y_test,
        error_y=dict(
            type='data', # value of error bar given in data coordinates
            array=y_test_err,
            visible=True)
    )],
    layout= dict(
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = 1
        
    ),
    yaxis = dict(
        
    ),
    xaxis_title = 'No. of fully-connected layers',
    yaxis_title = 'L1 Loss',
    legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="right",
    x=0.99
    ) 
    )
    )
    fig.show() 