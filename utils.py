# Created by alexandra at 20/12/2023
from math import exp
import torch as t

def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


def accuracy_fn(y_true, y_pred):
    correct = t.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc
