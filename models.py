"""Discriminator model for ADDA."""

from torch import nn
import torch


def get_model(name, n_outputs):
    if name == "lenet":
        model = EmbeddingNet(n_outputs).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9))

        return model.cuda(), opt

    if name == "disc":
        model = Discriminator(input_dims=500, hidden_dims=500, output_dims=2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9))

        return model.cuda(), opt


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, output_dims))

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


class EmbeddingNet(nn.Module):
    def __init__(self, n_outputs=128):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(
            # 1st conv layer
            # input [1 x 28 x 28]
            # output [20 x 12 x 12]
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # 2nd conv layer
            # input [20 x 12 x 12]
            # output [50 x 4 x 4]
            nn.Conv2d(20, 50, kernel_size=5),
            #nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU())
        self.n_classes = 10
        self.n_outputs = n_outputs
        self.fc = nn.Sequential(
            nn.Linear(50 * 4 * 4, 500),
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            nn.Linear(500, self.n_outputs))

    def extract_features(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc[0](output)
        return output

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)
