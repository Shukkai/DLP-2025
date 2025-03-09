import numpy as np

class module():
    def __init__(self)->None:
        super().__init__()
    def forward(self, input)->np.ndarray: 
        self.input = input
        return self.input
    def backward(self,grad)->np.ndarray:
        self.grad = grad
        return self.grad

class ReLU(module):
    def __init__(self)->None:
        super().__init__()
    def forward(self, input)->np.ndarray:
        self.input = input
        return np.maximum(0,input)
    def backward(self, grad)->np.ndarray:
        return (self.input > 0) * grad

class Sigmoid(module):
    def __init__(self)->None:
        super().__init__()
        self.grad = 0.0
    def forward(self, input)->np.ndarray:
        self.input = input
        self.grad = 1.0 / (1.0 + np.exp(-input))
        return self.grad
    def backward(self, grad)->np.ndarray:
        return grad  * (1 - self.grad) * self.grad

class Linear_layer_wo_active(module):
    def __init__(self, input_size, output_size, optim="sgd",lr=0.001, epsilon=1e-8, beta=0.9) -> None:
        super().__init__()
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)  # He Initialization
        self.bias = np.zeros((1, output_size))  # Initialize bias with zeros
        self.optim = optim
        self.lr = lr
        self.epsilon = epsilon
        self.beta = beta  # Momentum factor
        self.G = np.zeros_like(self.weights)
        self.G_bias = np.zeros_like(self.bias)
        self.v_w = np.zeros_like(self.weights)  # Velocity for weights
        self.v_b = np.zeros_like(self.bias)  # Velocity for bias
    def forward(self, input):
        self.input = input
        self.grad = np.dot(self.input, self.weights)  
        return self.grad

    def backward(self, grad):
        self.grad_weight = np.dot(self.input.T, grad)
        if self.optim == "sgd":
            self.weights -= self.lr * self.grad_weight
            self.bias -= self.lr * self.grad  # Update bias
        return np.dot(grad, self.weights.T)

class Linear_layer(module):
    def __init__(self, input_size, output_size, active="sigmoid", optim="sgd", lr=0.1, epsilon=1e-8, beta=0.9) -> None:
        super().__init__()
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)  # He Initialization
        self.bias = np.zeros((1, output_size))  # Initialize bias with zeros
        if active == "sigmoid":
            self.active = Sigmoid()
        elif active == "relu":
            self.active = ReLU()
        else:
            self.active = module()
        self.optim = optim
        self.lr = lr
        self.epsilon = epsilon
        self.beta = beta  # Momentum factor
        self.G = np.zeros_like(self.weights)
        self.G_bias = np.zeros_like(self.bias)
        self.v_w = np.zeros_like(self.weights)  # Velocity for weights
        self.v_b = np.zeros_like(self.bias)  # Velocity for bias

    def forward(self, input):
        self.input = input
        self.grad = self.active.forward(np.dot(self.input, self.weights) + self.bias)  # Apply bias
        return self.grad

    def backward(self, grad):
        grad_active = self.active.backward(grad)
        self.grad_weight = np.dot(self.input.T, grad_active)
        self.grad_bias = np.sum(grad_active, axis=0, keepdims=True)  # Compute bias gradient

        if self.optim == "sgd":
            self.weights -= self.lr * self.grad_weight
            self.bias -= self.lr * self.grad_bias  # Update bias

        elif self.optim == "mom":
            # Apply Momentum Update
            self.v_w = self.beta * self.v_w + (1 - self.beta) * self.grad_weight
            self.v_b = self.beta * self.v_b + (1 - self.beta) * self.grad_bias

            self.weights -= self.lr * self.v_w
            self.bias -= self.lr * self.v_b

        elif self.optim == "ada":
            self.G += self.grad_weight ** 2
            self.G_bias += self.grad_bias ** 2

            adjusted_lr = self.lr / (np.sqrt(self.G) + self.epsilon)
            adjusted_lr_bias = self.lr / (np.sqrt(self.G_bias) + self.epsilon)

            self.weights -= adjusted_lr * self.grad_weight
            self.bias -= adjusted_lr_bias * self.grad_bias  # Update bias

        return np.dot(grad_active, self.weights.T)
    
class Conv1D(module):
    def __init__(self, in_shape, out_shape, kernel_size=2, stride=1, active="sigmoid", optim="sgd", lr=0.01, epsilon=1e-8, beta=0.9) -> None:
        super().__init__()
        self.weights = np.random.randn(out_shape, in_shape, kernel_size)
        self.stride = stride
        self.lr = lr
        self.epsilon = epsilon
        self.G = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        if active == "sigmoid":
            self.active = Sigmoid()
        elif active == "relu":
            self.active = ReLU()
        else:
            self.active = module()  # No activation if not specified
        self.bias = np.zeros((1, out_shape))  # Initialize bias with zeros
        self.optim = optim
        self.beta = beta  # Momentum factor
        self.G = np.zeros_like(self.weights)
        self.G_bias = np.zeros_like(self.bias)
        self.v_w = np.zeros_like(self.weights)  # Velocity for weights
        self.v_b = np.zeros_like(self.bias)  # Velocity for bias
    def forward(self, input):
        # Reshape input to have a channel dimension if it's 2D (single sample case)
        if len(input.shape) == 2:
            input = input.reshape(input.shape[0], 1, input.shape[1])  # (1, in_channels, input_length)

        self.input = input
        output_length = input.shape[2] - self.weights.shape[2] + 1
        print(input)
        print((input.shape[2], self.weights.shape[2]))
        output = np.zeros((input.shape[0], self.weights.shape[0], output_length)) 

        # Convolution operation
        for i in range(output_length):
            for j in range(self.weights.shape[0]):  # Iterate over output channels
                output[:, j, i] = np.sum(input[:, :, i:i + self.weights.shape[2]] * self.weights[j], axis=(1, 2))

        # Apply activation function
        self.output = self.active.forward(output)
        return self.output

    def backward(self, gradwrtoutput):
        # Backpropagate the gradients with respect to the output
        grad_active = self.active.backward(gradwrtoutput)
        out_channels, output_length = grad_active.shape
        in_channels, _, kernel_size = self.weights.shape
        
        # Initialize gradients
        dX = np.zeros_like(self.input)
        self.gradW = np.zeros_like(self.W)

        # Gradient computation for the input and weights
        for i in range(output_length):
            for j in range(in_channels):
                for k in range(out_channels):
                    flipped_kernel = self.weights[k, j, ::-1]
                    valid_slice = slice(max(0, i * self.stride), min(self.input.shape[2], i * self.stride + kernel_size))
                    dX[:, j, valid_slice] += np.sum(gradwrtoutput[:, k, i:i + 1] * flipped_kernel[:, j], axis=0)

        for i in range(out_channels):
            for j in range(in_channels):
                for k in range(kernel_size):
                    self.gradW[i, j, k] = np.sum(gradwrtoutput[:, i, :] * np.roll(self.X[:, j, :], -k, axis=2), axis=0)

        # Apply the chosen optimizer
        if self.optim == "sgd":
            self.weights -= self.lr * self.gradW
        elif self.optim == "mom":
            self.v_w = self.beta * self.v_w + (1 - self.beta) * self.gradW
            self.weights -= self.lr * self.v_w
        elif self.optim == "ada":
            self.G += self.gradW ** 2
            adjusted_lr = self.lr / (np.sqrt(self.G) + self.epsilon)
            self.weights -= adjusted_lr * self.gradW

        return dX

