import numpy as np


class CEloss():
    def __init__(self,eps=1e-8):
        self.eps = eps
        self.ypred = None
        self.ytrue = None
    
    def CalLoss(self,y_true,y_pred):
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        self.ypred = y_pred
        self.ytrue = y_true
        row_wiseSummed = np.sum(y_true*np.log(y_pred),axis=1)
        loss = -np.mean(row_wiseSummed)
        return loss
    
    def backward(self):
        y_true = self.ytrue
        y_pred = self.ypred
        batch_size = y_true.shape[0]
        grad = -1*y_true / (y_pred * batch_size)
        return grad