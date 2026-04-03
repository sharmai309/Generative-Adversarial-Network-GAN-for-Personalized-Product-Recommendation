import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=128):
        super(Discriminator, self).__init__()
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, user_ids, item_ids):
        u = self.user_embed(user_ids)
        i = self.item_embed(item_ids)
        return self.net(torch.cat([u, i], dim=1))