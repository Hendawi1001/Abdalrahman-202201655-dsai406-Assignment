import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

batch_size = 256
epochs = 12
lr = 0.00004
z_dim = 50

df = pd.read_csv('mnist_train.csv')
images = df.iloc[:, 1:].values
images = (images.astype(np.float32) - 127.5) / 127.5
tensor_images = torch.tensor(images)
dataset = torch.utils.data.TensorDataset(tensor_images)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen = Generator().to(device)
disc = Discriminator().to(device)

opt_gen = optim.Adam(gen.parameters(), lr=lr)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
criterion = nn.BCELoss()

print(f"Training on {device}...")

for epoch in range(epochs):
    epoch_lossG = 0.0
    epoch_lossD = 0.0
    correct_disc = 0
    total_disc = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        real = batch[0].to(device)
        current_batch_size = real.shape[0]
        
        noise = torch.randn(current_batch_size, z_dim).to(device)
        fake = gen(noise)

        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        epoch_lossG += lossG.item()
        epoch_lossD += lossD.item()
        
        correct_disc += (disc_real > 0.5).sum().item() + (disc_fake < 0.5).sum().item()
        total_disc += 2 * current_batch_size

    avg_lossG = epoch_lossG / len(dataloader)
    avg_lossD = epoch_lossD / len(dataloader)
    acc_D = correct_disc / total_disc

    print(f"Epoch [{epoch+1}/{epochs}] | Loss D: {avg_lossD:.4f}, Loss G: {avg_lossG:.4f}, Acc D: {acc_D:.4f}")

print("Final Generator Loss:", lossG.item())
print("Final Discriminator Loss:", lossD.item())

noise = torch.randn(1, z_dim).to(device)
generated_img = gen(noise).cpu().detach().numpy().reshape(28, 28)

plt.imshow(generated_img, cmap='gray')
plt.title(f"Generated Image (LR: {lr}, Batch: {batch_size})")
img_path = "generated_sample.png"
plt.savefig(img_path)
plt.close()

torch.save(gen.state_dict(), "generator_model.pth")
torch.save(disc.state_dict(), "discriminator_model.pth")
    
print("Run complete!")