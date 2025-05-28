import numpy as np


class CEloss():
    def __init__(self,eps=1e-8):
        self.eps = eps
    
    def CalLoss(self,y_true,y_pred):
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        row_wiseSummed = np.sum(y_true*np.log(y_pred),axis=1)
        loss = -np.mean(row_wiseSummed)
        return loss
    
    def backward(self,y_true,y_pred):
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        batch_size = y_true.shape[0]
        grad = -y_true / (y_pred * batch_size)
        return grad