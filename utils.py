import torch
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

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
    writer.add_figure(message, plt.gcf(),e,True)