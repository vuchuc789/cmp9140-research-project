import torch.nn.init as init
from torch import nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        # If this is the output layer (size = 79), use Xavier
        if m.out_features == 79:
            init.xavier_uniform_(m.weight)
        else:
            init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            init.zeros_(m.bias)


class Autoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(79, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.3),  # let the latent space receive all information
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 79),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
