import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=128, noise_dim=64):
        super(Generator, self).__init__()
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim + noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_items),
            nn.Sigmoid()
        )

    def forward(self, user_ids):
        u = self.user_embed(user_ids)
        z = torch.randn(u.size(0), self.noise_dim, device=u.device)
        return self.net(torch.cat([u, z], dim=1))