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

class Tanh(module):
    def __init__(self):
        pass  # No need for initialization
    def forward(self, input: np.ndarray) -> np.ndarray:
        """Computes the Tanh activation function."""
        self.output = np.tanh(input)  # Store output for backpropagation
        return self.output
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Computes the derivative of Tanh activation."""
        return (1 - self.output ** 2) * grad  # Use stored output to compute derivative

class ReLU(module):
    def __init__(self)->None:
        super().__init__()
    def forward(self, input)->np.ndarray:
        self.input = input
        self.out = np.maximum(0, input)
        return np.maximum(0, input)
    def backward(self, grad)->np.ndarray:
        # Use the stored input to create the mask for the gradient
        return grad * (self.out > 0)

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
        # self.grad_bias = np.sum(grad, axis=0, keepdims=True)
        if self.optim == "sgd":
            self.weights -= self.lr * self.grad_weight
            # self.bias -= self.lr * self.grad_bias  # Update bias
        return np.dot(grad, self.weights.T)

class Linear_layer(module):
    def __init__(self, input_size, output_size, active="sigmoid", optim="sgd", lr=0.1, epsilon=1e-8, beta=0.9) -> None:
        super().__init__()
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        if active == "sigmoid":
            self.active = Sigmoid()
        elif active == "relu":
            self.active = ReLU()
        elif active == "tan":
            self.active = Tanh()
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
        self.grad = self.active.forward(np.dot(self.input, self.weights)+self.bias)  # Apply bias
        return self.grad
    def backward(self, grad):
        grad_active = self.active.backward(grad)
        self.grad_weight = np.dot(self.input.T, grad_active)
        self.grad_bias = np.sum(grad_active, axis=0, keepdims=True)  # Compute bias gradient
        if self.optim == "sgd":
            self.weights -= self.lr * self.grad_weight
            self.bias -= self.lr * self.grad_bias  # Update bias
        elif self.optim == "mom":
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
    
class Conv1D_layer(module):
    def __init__(self, input_length, num_filters, kernel_size, stride=1, padding=0,
                 active="sigmoid", optim="sgd", lr=0.1, epsilon=1e-8, beta=0.9) -> None:
        super().__init__()
        self.input_length = input_length
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # He initialization for filters. Shape: (num_filters, 1, kernel_size)
        self.weights = np.random.randn(num_filters, 1, kernel_size) * np.sqrt(2.0 / kernel_size)
        self.bias = np.zeros((1, num_filters))
        
        if active == "sigmoid":
            self.active = Sigmoid()
        elif active == "relu":
            self.active = ReLU()
        elif active == "tan":
            self.active = Tanh()
        else:
            self.active = module()
        self.optim = optim
        self.lr = lr
        self.epsilon = epsilon
        self.beta = beta
        self.G = np.zeros_like(self.weights)
        self.G_bias = np.zeros_like(self.bias)
        self.v_w = np.zeros_like(self.weights)
        self.v_b = np.zeros_like(self.bias)
        
    def forward(self, input):
        self.input = input  # (batch_size, input_length)
        batch_size = input.shape[0]
        input_reshaped = input.reshape(batch_size, 1, self.input_length)

        out_length = int((self.input_length - self.kernel_size + 2 * self.padding) / self.stride) + 1

        if self.padding > 0:
            self.input_padded = np.pad(input_reshaped, ((0,0), (0,0), (self.padding, self.padding)), mode='constant')
        else:
            self.input_padded = input_reshaped
        
        padded_length = self.input_padded.shape[2]
        self.pre_activation = np.zeros((batch_size, self.num_filters, out_length))
        
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(out_length):
                    start = i * self.stride
                    end = start + self.kernel_size
                    region = self.input_padded[b, 0, start:end]
                    self.pre_activation[b, f, i] = np.sum(region * self.weights[f, 0, :]) + self.bias[0, f]
                    
        self.out = self.active.forward(self.pre_activation)
        return self.out

    def backward(self, grad):
        grad_active = self.active.backward(grad)
        batch_size = self.input.shape[0]
        out_length = grad_active.shape[2]
        
        d_weights = np.zeros_like(self.weights)
        d_bias = np.zeros_like(self.bias)
        d_input_padded = np.zeros_like(self.input_padded)
        
        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(out_length):
                    start = i * self.stride
                    end = start + self.kernel_size
                    region = self.input_padded[b, 0, start:end]
                    
                    d_weights[f, 0, :] += grad_active[b, f, i] * region
                    d_bias[0, f] += grad_active[b, f, i]
                    d_input_padded[b, 0, start:end] += grad_active[b, f, i] * self.weights[f, 0, :]
        
        if self.padding > 0:
            d_input = d_input_padded[:, 0, self.padding:-self.padding]
        else:
            d_input = d_input_padded[:, 0, :]
        
        if self.optim == "sgd":
            self.weights -= self.lr * d_weights
            self.bias -= self.lr * d_bias
        elif self.optim == "mom":
            self.v_w = self.beta * self.v_w + (1 - self.beta) * d_weights
            self.v_b = self.beta * self.v_b + (1 - self.beta) * d_bias
            self.weights -= self.lr * self.v_w
            self.bias -= self.lr * self.v_b
        elif self.optim == "ada":
            self.G += d_weights ** 2
            self.G_bias += d_bias ** 2
            adjusted_lr = self.lr / (np.sqrt(self.G) + self.epsilon)
            adjusted_lr_bias = self.lr / (np.sqrt(self.G_bias) + self.epsilon)
            self.weights -= adjusted_lr * d_weights
            self.bias -= adjusted_lr_bias * d_bias
        
        return d_input
