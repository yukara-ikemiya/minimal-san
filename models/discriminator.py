import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common_layers import _Conv2d


class Discriminator(nn.Module):
    def __init__(self, num_class=10, model_type='san'):
        super(Discriminator, self).__init__()

        self.model_type = model_type
        self.layers = [128, 256, 512]

        self.conv0 = _Conv2d(1, self.layers[0], kernel_size=6, stride=2,
                             norm="none", activation="leaky_relu")
        self.conv1 = _Conv2d(self.layers[0], self.layers[1], kernel_size=4, stride=2, padding=1,
                             norm="spectral", activation="leaky_relu")
        self.conv2 = _Conv2d(self.layers[1], self.layers[2], kernel_size=4, stride=2, padding=1,
                             norm="spectral", activation="leaky_relu")

        self.use_class = num_class > 0
        # weights of Linear layer
        self.fc_w = nn.Parameter(torch.randn(num_class if self.use_class else 1, self.layers[2] * 3 * 3))

    def forward(self, x, class_ids, loss_type: str):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        h = torch.flatten(x, start_dim=1)  # (bs, self.layers[2] * 3 * 3)
        weights = self.fc_w[class_ids] if self.use_class else self.fc_w
        loss = self.compute_loss(h, weights, loss_type)

        return loss

    def compute_loss(self, x, weights, loss_type):
        """
        Hinge loss for discriminator training
        """
        assert (loss_type in ['G', 'D_real', 'D_fake'])

        if 'gan' == self.model_type:
            logits = (x * weights).sum(dim=1)  # (bs,)
            if 'G' == loss_type:
                loss = - logits.mean()
            elif 'D_real' == loss_type:
                loss = (1 - logits).relu().mean()
            else:  # 'D_fake
                loss = (1 + logits).relu().mean()
        elif 'san' == self.model_type:
            weights = F.normalize(weights, dim=1)
            if 'G' == loss_type:
                logits = (x * weights).sum(dim=1)
                loss = - logits.mean()

            elif 'D_real' == loss_type:
                loss_h = (1 - (x * weights.detach()).sum(dim=1)).relu().mean()
                loss_w = - (x.detach() * weights).sum(dim=1).mean()
                loss = (loss_h + loss_w) / 2.
            else:  # 'D_fake
                loss_h = (1 + (x * weights.detach()).sum(dim=1)).relu().mean()
                loss_w = (x.detach() * weights).sum(dim=1).mean()
                loss = (loss_h + loss_w) / 2.
        else:
            raise NotImplementedError

        return loss
