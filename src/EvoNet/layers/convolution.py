import torch
import torch.nn as nn


class Reducer(nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        out = 0
        for layer in self.layers:
            out = out + layer(x)
        return out
    
    def conjugate(self, x):
        out = 0
        for layer in self.layers:
            out = out + layer.conjugate(x)
        return out


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__()
        self.weights_real = nn.Parameter(torch.randn(in_features, out_features))
        self.weights_imag = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = bias
        if bias:
            self.bias_real = nn.Parameter(torch.randn(out_features))
            self.bias_imag = nn.Parameter(torch.randn(out_features))
        
    def forward(self, x):
        weights = self.weights_real - 1j * self.weights_imag
        out = x @ weights
        if self.bias:
            out += self.bias_real + 1j * self.bias_imag
        return out


class HarmonicConv2d(nn.Module):
    def __init__(self, in_features, out_features, template, dilations, bias=True) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.template = template
        self.layers = nn.ModuleList()
        self.dilations = dilations

        height, width = template.shape[-2:]
        for d in range(1, self.dilations + 1):
            dilated_height, dilated_width = d*(height-1)+1, d*(width-1)+1
            basis_count = (in_features[0] - dilated_height + 1) * (in_features[1] - dilated_width + 1)
            # self.layers.append(ComplexLinear(basis_count, out_features, bias=False))
            self.layers.append(nn.Linear(basis_count, out_features, bias=False))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        
        self.flatten = nn.Flatten()

    
    def forward(self, x):
        device = 'cuda' if next(self.parameters()).is_cuda else 'cpu'
        eigen = self.template * torch.pi
        out = 0 if self.bias is None else self.bias
        for d in range(1, self.dilations + 1):
            features = self.flatten(torch.cos(torch.conv2d(x, eigen, dilation=d))) / torch.linalg.norm(eigen)
            out = out + self.layers[d-1](features)
        
        return out
    
    def conjugate(self, x):
        device = 'cuda' if next(self.parameters()).is_cuda else 'cpu'
        eigen = self.template * torch.pi
        out = 0 if self.bias is None else self.bias
        for d in range(1, self.dilations + 1):
            features = self.flatten(torch.sin(torch.conv2d(x, eigen, dilation=d))) / torch.linalg.norm(eigen)
            out = out + self.layers[d-1](features)
        
        return out