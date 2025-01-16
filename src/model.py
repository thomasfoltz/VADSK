import torch.nn as nn

class VADSK(nn.Module):
    def __init__(self, feature_dim):
        super(VADSK, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x.T
    

class VADSK_AUTOENCODER(nn.Module):
    def __init__(self, feature_dim):
        super(VADSK_AUTOENCODER, self).__init__()
        # Encoder
        self.enc_fc1 = nn.Linear(feature_dim, 512)
        self.enc_relu1 = nn.ReLU()
        self.enc_fc2 = nn.Linear(512, 256)
        self.enc_relu2 = nn.ReLU()
        self.enc_fc3 = nn.Linear(256, 128)  # Latent space dimension

        # Decoder
        self.dec_fc1 = nn.Linear(128, 256)
        self.dec_relu1 = nn.ReLU()
        self.dec_fc2 = nn.Linear(256, 512)
        self.dec_relu2 = nn.ReLU()
        self.dec_fc3 = nn.Linear(512, feature_dim)  # Output matches input dimension

    def forward(self, x):
        # Encoder
        x = self.enc_relu1(self.enc_fc1(x))
        x = self.enc_relu2(self.enc_fc2(x))
        x = self.enc_fc3(x)  # Encoded representation

        # Decoder
        x = self.dec_relu1(self.dec_fc1(x))
        x = self.dec_relu2(self.dec_fc2(x))
        x = self.dec_fc3(x)  # Reconstructed input
        return x