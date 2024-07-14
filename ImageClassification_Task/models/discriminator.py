import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(128, 32, 3, stride=1, padding=1),          
            nn.ReLU()
        ) # input is [52, 128, 28, 28]
        
        self.decoder = nn.Sequential(
			nn.ConvTranspose2d(32, 128, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x