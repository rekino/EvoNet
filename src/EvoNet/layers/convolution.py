import torch
import torch.nn as nn

from itertools import product


class Reducer(nn.Module):
    def __init__(self, layers, activation=None) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.activation = activation
    
    def forward(self, x):
        out = 0
        for layer in self.layers:
            out = out + layer(x)
        return out if self.activation is None else self.activation(out)


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, real=False) -> None:
        super().__init__()
        self.real = real
        self.weights_real = nn.Parameter(torch.randn(in_features, out_features))
        self.weights_imag = torch.zeros_like(self.weights_real) if real else nn.Parameter(torch.randn(in_features, out_features))

        self.bias = bias
        if bias:
            self.bias_real = nn.Parameter(torch.randn(out_features))
            self.bias_imag = torch.zeros_like(self.bias_real) if real else nn.Parameter(torch.randn(out_features))
        
    def forward(self, x):
        weights = self.weights_real - 1j * self.weights_imag
        out = x @ weights
        if self.bias:
            out += self.bias_real + 1j * self.bias_imag
        return out


class HolomorphicConv2d(nn.Module):
    def __init__(self, in_features, out_features, template, dilations, bias=True, real=False) -> None:
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
            self.layers.append(ComplexLinear(basis_count, out_features, bias=False, real=real))
        
        if bias:
            self.bias_real = nn.Parameter(torch.randn(out_features))
            self.bias_imag = torch.zeros_like(self.bias_real) if real else nn.Parameter(torch.randn(out_features))
        
        self.flatten = nn.Flatten()

    
    def forward(self, x):
        # device = 'cuda' if next(self.parameters()).is_cuda else 'cpu'
        eigen = self.template * torch.pi
        out = 0 if self.bias_real is None else (self.bias_real + 1j * self.bias_imag)
        for d in range(1, self.dilations + 1):
            features = self.flatten(torch.exp(1j * torch.conv2d(x, eigen, dilation=d))) / torch.linalg.norm(eigen)
            out = out + self.layers[d-1](features)
        
        return out


class HarmonicConv2d(nn.Module):
    def __init__(self, in_features, out_features, template, dilations, bias=True) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        
        self.layers = nn.ModuleList()
        self.dilations = dilations

        height, width = template.shape[-2:]
        for d in range(1, dilations + 1):
            dilated_height, dilated_width = d*(height-1)+1, d*(width-1)+1
            basis_count = (in_features[0] - dilated_height + 1) * (in_features[1] - dilated_width + 1)
            self.layers.append(nn.Linear(basis_count, out_features, bias=False))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        
        self.flatten = nn.Flatten()

        self.degree = torch.sum(template, dtype=torch.int).item()
        indices = torch.nonzero(template, as_tuple=True)
        self.template = torch.zeros(2**self.degree, *template.shape[1:])
        for i, flip in enumerate(product([1, -1], repeat=self.degree)):
            temp = template
            temp[indices] = temp[indices] * torch.asarray(flip)
            self.template[i] = temp[0]

    def forward(self, x):
        # device = 'cuda' if next(self.parameters()).is_cuda else 'cpu'
        eigen = self.template * torch.pi
        out = 0 if self.bias is None else self.bias
        for d in range(1, self.dilations + 1):
            temp = torch.cos(torch.conv2d(x, eigen, dilation=d))
            features = self.flatten(torch.sum(temp, dim=1, keepdim=True) / 2**self.degree)
            out = out + self.layers[d-1](features / torch.linalg.norm(eigen))
        
        return out