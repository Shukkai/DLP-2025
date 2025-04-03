import numpy as np
from utils import Linear_layer, Linear_layer_wo_active, Conv1D_layer
class Linear_wo_active_Model:
    def __init__(self, X, y, hidden_size=4):
        super().__init__()
        self.X = X
        self.y = y
        self.layer1 =  Linear_layer_wo_active(self.X.shape[1],hidden_size)
        self.layer2 =  Linear_layer_wo_active(hidden_size,hidden_size)
        self.layer3 =  Linear_layer_wo_active(hidden_size,1)

    def forward(self, input):
        self.w1 = self.layer1.forward(input)
        self.w2 = self.layer2.forward(self.w1)
        self.w3 = self.layer3.forward(self.w2)
        return self.w3
    def backward(self):
        y = self.forward(self.X)
        loss = np.mean((y - self.y) ** 2)
        dloss = 2*(y-self.y) #derivative
        #backpropagration
        dw3 = self.layer3.backward(dloss)
        dw2 = self.layer2.backward(dw3)
        dw1 = self.layer1.backward(dw2)
        return loss
    
class Linear_Model:
    def __init__(self, X, y, hidden_size=4, lr=0.001,activate = "sigmoid",optim = "sgd"):
        super().__init__()
        self.X = X
        self.y = y
        self.layer1 =  Linear_layer(self.X.shape[1],hidden_size, activate, optim)
        self.layer2 =  Linear_layer(hidden_size,hidden_size, "sigmoid", optim)
        self.layer3 =  Linear_layer(hidden_size,1, "sigmoid", optim) # flatten
        self.lr = lr
    def forward(self, input):
        self.w1 = self.layer1.forward(input)
        self.w2 = self.layer2.forward(self.w1)
        self.w3 = self.layer3.forward(self.w2)
        return self.w3
    def backward(self):
        y = self.forward(self.X)
        loss = np.mean((y - self.y) ** 2) # gt - pred
        dloss = 2*(y - self.y) #derivative
        #backpropagration
        dw3 = self.layer3.backward(dloss)
        dw2 = self.layer2.backward(dw3)
        dw1 = self.layer1.backward(dw2)
        return loss

class Conv_Model:
    def __init__(self, X, y, conv_filters=2, lr=0.001, activate_conv="relu", activate_linear="sigmoid", optim="sgd"):
        super().__init__()
        self.X = X  # Expecting shape (batch_size, 2)
        self.y = y  # Expecting shape (batch_size, 1)
        # Conv layer: input_length=2, kernel_size=2 produces output shape (batch_size, conv_filters, 1)
        self.layer1 = Conv1D_layer(input_length=self.X.shape[1],
                                    num_filters=conv_filters,
                                    kernel_size=2,
                                    stride=1,
                                    padding=0,
                                    active=activate_conv,
                                    optim=optim,
                                    lr=lr)
        # Flatten the conv output: from (batch_size, conv_filters, 1) to (batch_size, conv_filters)
        self.layer2 = Linear_layer(conv_filters, conv_filters, active=activate_linear, optim=optim, lr=lr)
        self.layer3 = Linear_layer(conv_filters, 1, active=activate_linear, optim=optim, lr=lr)

    def forward(self, input):
        # input shape: (batch_size, 2)
        conv_out = self.layer1.forward(input)  # shape: (batch_size, conv_filters, 1)
        flat = conv_out.reshape(conv_out.shape[0], -1)  # shape: (batch_size, conv_filters)
        out2 = self.layer2.forward(flat)
        out3 = self.layer3.forward(out2)
        return out3

    def backward(self):
        y_pred = self.forward(self.X)
        loss = np.mean((y_pred - self.y) ** 2)
        dloss = 2 * (y_pred - self.y)
        grad3 = self.layer3.backward(dloss)  # shape: (batch_size, conv_filters)
        grad2 = self.layer2.backward(grad3)    # shape: (batch_size, conv_filters)
        # Reshape gradient to match conv layer's output shape: (batch_size, conv_filters, 1)
        grad2_reshaped = grad2.reshape(self.layer1.out.shape)
        grad1 = self.layer1.backward(grad2_reshaped)
        return loss
