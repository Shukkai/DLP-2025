import numpy as np
from utils import Linear_layer, Conv1D, Linear_layer_wo_active

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
    def __init__(self, X, y, hidden_size=10, lr=0.001,activate = "sigmoid",optim = "sgd"):
        super().__init__()
        self.X = X
        self.y = y
        self.layer1 =  Linear_layer(self.X.shape[1],hidden_size, activate, optim)
        self.layer2 =  Linear_layer(hidden_size,hidden_size, activate, optim)
        self.layer3 =  Linear_layer(hidden_size,1, activate, optim) # flatten
        self.lr = lr
    def forward(self, input):
        self.w1 = self.layer1.forward(input)
        self.w2 = self.layer2.forward(self.w1)
        self.w3 = self.layer3.forward(self.w2)
        return self.w3
    def backward(self):
        y = self.forward(self.X)
        loss = np.mean((self.y - y) ** 2)
        dloss = 2*(y-self.y) #derivative
        #backpropagration
        dw3 = self.layer3.backward(dloss)
        dw2 = self.layer2.backward(dw3)
        dw1 = self.layer1.backward(dw2)
        return loss

class Conv_Model:
    def __init__(self, X, y, hidden_size = 10,lr=0.001, activate="sigmoid", optim="sgd"):
        super().__init__()
        self.X = X
        self.y = y
        # Initialize convolutional layers
        self.layer1 = Linear_layer(input_size=X.shape[1], output_size=1, active=activate, optim=optim, lr=lr)
        self.layer2 = Conv1D(in_shape=hidden_size, out_shape=hidden_size, kernel_size=2, stride=1, active=activate, optim=optim, lr=lr)
        # The final layer is a Linear layer to output a scalar (regression task)
        self.layer3 = Linear_layer(input_size=hidden_size, output_size=1, active=activate, optim=optim, lr=lr)
        self.lr = lr
    
    def forward(self, input):
        input = input.reshape(input.shape[0], 2, 1)  # Reshape to (batch_size, 2, 1)
        # Pass the input through the convolutional layers
        out1 = self.layer1.forward(input)  # Output shape (batch_size, 10, output_length)
        out2 = self.layer2.forward(out1)   # Output shape (batch_size, 10, output_length)
        # Flatten the output of the convolutional layers before passing to the linear layer
        out2_flat = out2.flatten(axis=1)  # Flatten (batch_size, 10, output_length) to (batch_size, 10*output_length)
        # Final linear layer to produce the output (1 output per batch)
        out3 = self.layer3.forward(out2_flat)
        return out3
    
    def backward(self):
        # Forward pass to get predictions
        y_pred = self.forward(self.X)
        # Calculate loss (Mean Squared Error)
        loss = np.mean((self.y - y_pred) ** 2)
        dloss = 2 * (y_pred - self.y)  # Derivative of MSE loss
        # Backpropagation through layers
        dw3 = self.layer3.backward(dloss)  # Backward through Linear layer
        dw2 = self.layer2.backward(dw3)    # Backward through 2nd Conv layer
        dw1 = self.layer1.backward(dw2)    # Backward through 1st Conv layer
        
        return loss
