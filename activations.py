import numpy as np

class ReLU():
    def __init__(self):
        self.out_vector = None
    def forward(self,input_vector):
        out_vec = np.where(input_vector>0,input_vector,0)
        self.out_vector = out_vec
        return out_vec
    def backward(self):
        gradient = np.where(self.out_vector>0,1,0)
        return gradient


class Sigmoid():
    def __init__(self):
        self.out_vec = None
    def forward(self,in_vec):
        in_vec = np.clip(in_vec,-500,500)
        out_vec = 1/(1+np.exp(-in_vec))
        self.out_vec = out_vec
        return out_vec
    def backward(self):
        gradient = self.out_vec*(1-self.out_vec)
        return gradient

class Tanh():
    def __init__(self):
        self.out_vec = None
    def forward(self,in_vec):
        in_vec = np.clip(in_vec,-500,500)
        out_vec = np.tanh(in_vec)
        self.out_vec = out_vec
        return out_vec
    def backward(self):
        gradient = 1-np.square(self.out_vec)
        return gradient
    
class Softmax():
    def __init__(self):
        self.out_vec = None
    
    def forward(self,x):
        #softmax is rowwise
        x = x-np.max(x,axis=1,keepdims=True)
        expX = np.exp(x)
        expSum = np.sum(expX,axis=1,keepdims=True)
        self.out_vec = expX/expSum
        return self.out_vec
    
    def backward(self):
        #Jacobians = tensor of shape batch_size,num_classes,num_classes
        Jacobians = np.zeros(shape=(self.out_vec.shape[0],self.out_vec.shape[1],self.out_vec.shape[1]))
        # mask with ones everywhere except on the diagonal. its basicall a matrix of 1s - a diagonal matrix of 1s. eye is a numpy function that creates such a matrix.
        mask = np.ones(shape=(self.out_vec.shape[1],self.out_vec.shape[1])) - np.eye(self.out_vec.shape[1],self.out_vec.shape[1])
        for x in range(self.out_vec.shape[0]):
            off_diagonal = np.outer(self.out_vec[x],-self.out_vec[x])
            off_diagonal = off_diagonal*mask
            diagonal = np.diag(self.out_vec[x]*(1-self.out_vec[x]))
            Jacobian = off_diagonal + diagonal
            Jacobians[x] = Jacobian
        return Jacobians

