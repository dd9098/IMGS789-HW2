# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
latent_dim = 100
img_size = 28 * 28
batch_size = 128
learning_rate = 0.0002
num_epochs = 100
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device: ", device)

# Data Loading
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Generator Model
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize models
generator = Generator(latent_dim, img_size).to(device)
discriminator = Discriminator(img_size).to(device)

# Loss and Optimizers
criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Utility function to denormalize images
def denormalize(img):
    img = img * 0.5 + 0.5
    return img

# Training Loop
g_losses = []
d_losses = []

for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # Flatten the images and move to device
        real_imgs = imgs.view(-1, img_size).to(device)
        batch_size = real_imgs.size(0)

        # Labels for real and fake images
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = generator(z)

        real_loss = criterion(discriminator(real_imgs), real_labels)
        fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)
        d_loss = real_loss + fake_loss

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = generator(z)
        g_loss = criterion(discriminator(fake_imgs), real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Save losses for plotting
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

    # Print progress
    print(f"Epoch [{epoch+1}/{num_epochs}]  D Loss: {d_loss.item():.4f}  G Loss: {g_loss.item():.4f}")

    # Visualize the generated images at certain epochs
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        with torch.no_grad():
            z = torch.randn(16, latent_dim).to(device)
            sample_imgs = generator(z).view(-1, 1, 28, 28)
            grid = torchvision.utils.make_grid(sample_imgs, nrow=4, normalize=True)
            plt.figure(figsize=(4, 4))
            plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
            plt.title(f"Generated Images at Epoch {epoch+1}")
            plt.axis("off")
            plt.show()

# Latent Space Interpolation for MNIST Dataset
z_start = torch.randn(1, latent_dim).to(device)
z_end = torch.randn(1, latent_dim).to(device)

# Generate interpolation steps
num_interpolations = 10
interpolated_images = []

for alpha in np.linspace(0, 1, num_interpolations):
    z = (1 - alpha) * z_start + alpha * z_end
    with torch.no_grad():
        interpolated_image = generator(z).view(1, 1, 28, 28).cpu()
        interpolated_images.append(interpolated_image)

# Plot interpolated images
plt.figure(figsize=(20, 4))
for i, img in enumerate(interpolated_images):
    plt.subplot(1, num_interpolations, i + 1)
    plt.imshow(img.squeeze().numpy(), cmap='gray')
    plt.axis("off")
    plt.title(f"Step {i+1}")
plt.suptitle("Latent Space Interpolation Between Two Images (MNIST)")
plt.show()

# Comment on the smoothness of the interpolation
# The latent space interpolation shows a smooth transition between two generated images. 
# The quality of the transitions can indicate how well the generator has learned the underlying data distribution. 
# Ideally, the intermediate images should look realistic and transition gradually without abrupt changes, 
# which reflects a well-structured latent space.

# Plot the loss curves
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label="Generator Loss")
plt.plot(d_losses, label="Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.title("Generator and Discriminator Loss During Training")
plt.show()
