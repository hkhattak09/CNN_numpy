import numpy as np

class Linear():
    def __init__(self,in_nodes,out_nodes,lr=0.001):
        self.in_ = in_nodes
        self.out_ = out_nodes
        # self.weight_ = np.random.normal(size=(in_nodes,out_nodes))
        fan_in_linear = in_nodes
        std_dev_linear = np.sqrt(2.0 / fan_in_linear)
        self.weight_ = np.random.normal(loc=0, scale=std_dev_linear, size=(in_nodes,out_nodes))
        self.bias = np.random.normal(size=(1,out_nodes))
        self.input = None
        self.output = None
        self.lr = lr
    
    def forward(self,input_vector):
        self.input = input_vector
        #output is data@Weight + bias.
        self.output = (input_vector @ self.weight_) + self.bias
        return self.output
    
    def  backward(self, error_signal):
        dw = self.input.T @ error_signal
        db = np.sum(error_signal,axis=0,keepdims=True)
        error_signal = error_signal @ self.weight_.T
        #updates
        self.weight_ = self.weight_ - self.lr*dw
        self.bias = self.bias - self.lr*db
        return error_signal


#Conv2d is being designed to work with square and rectangular matrices.
class Conv2d():
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding=0,lr=0.01):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        # self.kernels = np.random.normal(size=(out_channels,in_channels,kernel_size,kernel_size))
        fan_in_conv = in_channels * kernel_size * kernel_size
        std_dev_conv = np.sqrt(2.0 / fan_in_conv)
        self.kernels = np.random.normal(loc=0, scale=std_dev_conv, size=(out_channels,in_channels,kernel_size,kernel_size))
        self.bias = np.random.normal(size=(out_channels,1,1))
        self.input = None
        self.output = None
        self.lr = lr

    def convolve(self,input_data,kernels,padding,stride,out_channels,bias=False):
        # #conversion to int as np.floor returns a float while indexing requires int.
        out_H = int(np.floor(((input_data.shape[2] + 2*padding - kernels.shape[2])/stride)+1))
        out_W = int(np.floor(((input_data.shape[3] + 2*padding - kernels.shape[2])/stride)+1))
        output_tensor = np.zeros(shape=(input_data.shape[0],out_channels,out_H,out_W))
        if padding != 0:
            tensor = np.pad(input_data,pad_width=((0,0),(0,0),(padding,padding),(padding,padding)),mode='constant',constant_values=0)
        else:
            tensor = input_data
        #travel path is (0,0)->till right_edge->to(kernel_size,0)->right_edge and repeat.
        for batch_id in range(len(tensor)):
            for f in range(out_channels):
                for i in range(out_H):
                    for j in range(out_W):
                        #iterators represent the starting point of the kernel on the input matrix.
                        matrix = kernels[f,:,:,:] * tensor[batch_id,:,i*stride:i*stride+kernels.shape[2],j*stride:j*stride+kernels.shape[2]]
                        output_tensor[batch_id,f,i,j] = np.sum(matrix)
            if bias is True:
                output_tensor[batch_id,:,:,:] = output_tensor[batch_id,:,:,:] + self.bias
        return output_tensor
    

    def forward(self,input_data):
        # #input data is Batch_size,In_channels,H,W
        self.input = input_data
        self.output = self.convolve(input_data,self.kernels,self.padding,self.stride,self.out_channels,bias=True)
        return self.output
    
    def upsample(self,signal):
        H_out = signal.shape[2]
        W_out = signal.shape[3]
        H_upsampled = (H_out - 1) * self.stride + 1
        W_upsampled = (W_out - 1) * self.stride + 1
        upsampled_error_signal = np.zeros((signal.shape[0],signal.shape[1],H_upsampled,W_upsampled), dtype=signal.dtype)
        upsampled_error_signal[:,:,::self.stride,::self.stride] = signal
        return upsampled_error_signal


    
    def backward(self,error_signal):
        #error signal is a [batch_size,out_channels,out_dim,out_dim] what you basically have is pd of E wrt to O
        #computing the error signal to back propogate
        padding_dim = self.kernel_size - 1
        flipped_kernels = (self.kernels[:,:,::-1,::-1]).transpose(1,0,2,3)
        if self.stride>1:
            dXerror_signal = self.upsample(error_signal)
        else:
            dXerror_signal = error_signal

        new_errorsignal = self.convolve(dXerror_signal,flipped_kernels,padding=padding_dim,stride=1,out_channels=self.in_channels)
        #weight_update.
        #dw

        out_H = error_signal.shape[2]
        out_W = error_signal.shape[3]

        if self.padding != 0:
            tensor = np.pad(self.input,pad_width=((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),mode='constant',constant_values=0)
        else:
            tensor = self.input

        dw = np.zeros(shape=(self.out_channels,self.in_channels,self.kernel_size,self.kernel_size))
        #error signal is B,O,H,W
        #tensor is input and is B,I,H,W
        #dw is O,I,H,W as kernel is O,I,H,W
        for batch_id in range(error_signal.shape[0]):
            for out_filter in range(error_signal.shape[1]):
                for i in range(out_H):
                    for j in range(out_W):
                        output_map = error_signal[batch_id,out_filter,i,j]
                        matrix = output_map*tensor[batch_id,:,i*self.stride:i*self.stride+self.kernel_size,j*self.stride:j*self.stride+self.kernel_size]
                        dw[out_filter,:,:,:] = dw[out_filter,:,:,:] + matrix
        
        dw = dw/tensor.shape[0]

        db = np.sum(error_signal,axis=(0,2,3),keepdims=True)
        self.kernels = self.kernels - self.lr*dw
        self.bias = self.bias - self.lr*db

        return new_errorsignal


class Flatten():
    def __init__(self):
        self.input_dim = None
    
    def forward(self,tensor):
        #tensor shape is B,C,H,W
        #output shape is B,C*H*W
        self.input_dim = (tensor.shape[1],tensor.shape[2],tensor.shape[3])
        matrix = tensor.reshape(tensor.shape[0],-1)
        return matrix
    
    def backwards(self,matrix):
        tensor = matrix.reshape(matrix.shape[0],self.input_dim[0],self.input_dim[1],self.input_dim[2])
        return tensor
