import torch.nn as nn


def get_activation(activation, **kwargs):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "swish":
        return nn.SiLU()
    else:
        return nn.Identity()


class _Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, norm="batch", activation="relu", **kwargs):
        super(_Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

        if norm == "spectral":
            self.conv = nn.utils.spectral_norm(self.conv)
            self.norm = None
        elif norm == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None

        self.activation = get_activation(activation)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        out = self.activation(x)

        return out


class _ConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, norm="batch", activation="relu", **kwargs):
        super(_ConvTranspose2d, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)

        if norm == "spectral":
            self.conv = nn.utils.spectral_norm(self.conv)
            self.norm = None
        elif norm == "batch":
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None

        self.activation = get_activation(activation)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        out = self.activation(x)

        return out
