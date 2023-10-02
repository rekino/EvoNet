import torch.nn as nn
import torch


class SphericalLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, eps=1e-8) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = torch.tensor(eps)

        self.weights = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x):
        inner = x @ self.weights
        weights_norm = torch.linalg.norm(self.weights, dim=0, keepdims=True)
        x_norm = torch.linalg.norm(x, dim=-1, keepdims=True)
        denom = torch.maximum(x_norm * weights_norm, self.eps)
        cosine_similarity = inner / denom

        return torch.hstack([x_norm, cosine_similarity])
