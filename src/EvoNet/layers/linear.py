import torch.nn as nn
import torch


class SphericalLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight_count = in_features * (in_features - 1) // 2

        self.weights = nn.Parameter(torch.randn(out_features, weight_count))
