import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import MovieLensDataset
from src.generator import Generator
from src.discriminator import Discriminator
from tqdm import tqdm

def train(epochs=50, batch_size=64, embed_dim=128, lr=0.0002):
    dataset = MovieLensDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    G = Generator(dataset.n_users, dataset.n_items, embed_dim)
    D = Discriminator(dataset.n_users, dataset.n_items, embed_dim)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        g_losses, d_losses = [], []
        for users, items, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # Train Discriminator
            real_scores = D(users, items)
            real_loss = criterion(real_scores, torch.ones_like(real_scores))
            fake_items = G(users).argmax(dim=1)
            fake_scores = D(users, fake_items)
            fake_loss = criterion(fake_scores, torch.zeros_like(fake_scores))
            d_loss = (real_loss + fake_loss) / 2
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            # Train Generator
            fake_items = G(users).argmax(dim=1)
            fake_scores = D(users, fake_items)
            g_loss = criterion(fake_scores, torch.ones_like(fake_scores))
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        print(f"Epoch {epoch+1}: G={sum(g_losses)/len(g_losses):.4f}  D={sum(d_losses)/len(d_losses):.4f}")

    torch.save(G.state_dict(), 'models/generator.pt')
    torch.save(D.state_dict(), 'models/discriminator.pt')
    print("Models saved!")
    return G, D, dataset

if __name__ == '__main__':
    train()