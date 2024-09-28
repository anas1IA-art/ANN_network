
import torch.nn as nn
import numpy as np
import torch

__all__ = ['NeuralNetwork',]


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid', init_method='xavier'):
        super(NeuralNetwork, self).__init__()
        self.init_method = init_method
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError("Unsupported activation function")
        
        self.init_weights(init_method)
    
    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return torch.sigmoid(x)
    
    def init_weights(self, method):
        if method == 'zeros':
            nn.init.zeros_(self.hidden.weight)
            nn.init.zeros_(self.output.weight)
        elif method == 'ones':
            nn.init.ones_(self.hidden.weight)
            nn.init.ones_(self.output.weight)
        elif method == 'uniform':
            nn.init.uniform_(self.hidden.weight, -1/np.sqrt(self.hidden.in_features), 1/np.sqrt(self.hidden.in_features))
            nn.init.uniform_(self.output.weight, -1/np.sqrt(self.output.in_features), 1/np.sqrt(self.output.in_features))
        elif method == 'normal':
            nn.init.normal_(self.hidden.weight, std=0.01)
            nn.init.normal_(self.output.weight, std=0.01)
        elif method == 'xavier':
            nn.init.xavier_uniform_(self.hidden.weight)
            nn.init.xavier_uniform_(self.output.weight)
        elif method == 'he':
            nn.init.kaiming_uniform_(self.hidden.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.output.weight, nonlinearity='relu')
        elif method == 'uniform_scaled':
            self._init_uniform_scaled(self.hidden)
            self._init_uniform_scaled(self.output)
        else:
            raise ValueError("Unsupported initialization method")

    def _init_uniform_scaled(self, layer):
        n = layer.in_features
        y = 1.0 / np.sqrt(n)
        nn.init.uniform_(layer.weight, -y, y)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    