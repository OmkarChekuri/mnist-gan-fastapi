import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import logging
import time
import matplotlib.pyplot as plt
import numpy as np
import yaml
import imageio
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont

# --- 1. Logging Setup ---
def setup_logging(log_file):
    """Sets up logging to both console and a file."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# --- 2. Configuration Parameters ---
class GANConfig:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.latent_dim = config['latent_dim']
        self.img_shape = tuple(config['img_shape'])
        self.lr = config['lr']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


# --- 3. Model Architecture ---
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.size(0), *self.img_shape)
        return img
    
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

# --- 4. Training Script for Generator ---
def train_and_save_generator(output_path, logger, config, run_timestamp):
    """
    Trains a GAN Generator on the real MNIST dataset and saves the model state.
    """
    logger.info("--- Starting Model Training and Export ---")
    
    device = config.device
    latent_dim = config.latent_dim
    img_shape = config.img_shape
    lr = config.lr
    epochs = config.epochs
    batch_size = config.batch_size
    
    os.makedirs('data/mnist', exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    mnist_dataset = datasets.MNIST(root='data/mnist', train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist_dataset, batch_size=batch_size)

    generator = Generator(latent_dim, img_shape).to(device)
    discriminator = Discriminator(img_shape).to(device)
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    generator.train()
    discriminator.train()
    
    g_losses_per_epoch = []
    d_losses_per_epoch = []
    frames = []
    
    fixed_noise = torch.randn(64, latent_dim, device=device)

    for epoch in range(epochs):
        g_loss_for_epoch = 0.0
        d_loss_for_epoch = 0.0
        for i, (imgs, _) in enumerate(dataloader):
            
            # --- Train Discriminator ---
            optimizer_d.zero_grad()
            real_imgs = imgs.to(device)
            real_validity = discriminator(real_imgs)
            real_loss = criterion(real_validity, torch.ones(imgs.size(0), 1).to(device))

            z = torch.randn(imgs.size(0), latent_dim, device=device)
            fake_imgs = generator(z).detach()
            fake_validity = discriminator(fake_imgs)
            fake_loss = criterion(fake_validity, torch.zeros(imgs.size(0), 1).to(device))

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()
            
            # --- Train Generator ---
            optimizer_g.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim, device=device)
            gen_imgs = generator(z)
            g_loss = criterion(discriminator(gen_imgs), torch.ones(imgs.size(0), 1).to(device))
            
            g_loss.backward()
            optimizer_g.step()

            if i % 100 == 0:
                logger.info(f"Epoch: {epoch+1}/{epochs}, Batch: {i}/{len(dataloader)} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
            
            g_loss_for_epoch += g_loss.item()
            d_loss_for_epoch += d_loss.item()
        
        avg_g_loss = g_loss_for_epoch / len(dataloader)
        avg_d_loss = d_loss_for_epoch / len(dataloader)

        logger.info(f"Epoch {epoch+1}/{epochs} | D Loss (Avg): {avg_d_loss:.4f} | G Loss (Avg): {avg_g_loss:.4f}")
        g_losses_per_epoch.append(avg_g_loss)
        d_losses_per_epoch.append(avg_d_loss)

        with torch.no_grad():
            generator.eval()
            gen_imgs = generator(fixed_noise).detach().cpu()

            grid_img_tensor = vutils.make_grid(gen_imgs * 0.5 + 0.5, padding=2, normalize=True)
            grid_img_np = grid_img_tensor.permute(1, 2, 0).numpy()
            grid_img_pil = Image.fromarray((grid_img_np * 255.0).astype(np.uint8))
            
            draw = ImageDraw.Draw(grid_img_pil)
            font_size = 20
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
            
            text = f"Epoch: {epoch+1}"
            draw.text((10, 10), text, font=font, fill=(255, 255, 255))
            
            frames.append(np.array(grid_img_pil))
            generator.train()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(generator.state_dict(), output_path)
    logger.info(f"\nGenerator model successfully saved to {output_path}")

    # --- Plotting Training Metrics ---
    def visualize_training(g_losses, d_losses, save_path):
        """Generates and saves a plot of Generator and Discriminator losses."""
        epochs = np.arange(1, len(g_losses) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(epochs, g_losses, label="Generator Loss")
        plt.plot(epochs, d_losses, label="Discriminator Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Loss plot saved to {save_path}")

    plot_save_path = os.path.join('logs', f'gan_training_losses_{run_timestamp}.png')
    visualize_training(g_losses_per_epoch, d_losses_per_epoch, plot_save_path)
    
    # --- Generate GIF with Timestamp in filename ---
    gif_path = os.path.join('logs', f'gan_training_progress_{run_timestamp}.gif')
    logger.info("Creating GIF of generated samples over epochs...")
    imageio.mimsave(gif_path, frames, fps=1)
    logger.info(f"GIF created: {gif_path}")

    logger.info(f"--- Epoch Metrics Log ---")
    logger.info(f"Generator Losses: {g_losses_per_epoch}")
    logger.info(f"Discriminator Losses: {d_losses_per_epoch}")
    logger.info("--- Model Export Complete ---")


if __name__ == '__main__':
    model_save_path = os.path.join('model', 'generator_model.pth')
    
    if not os.path.exists('logs'):
        os.makedirs('logs')
    run_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    log_file_name = f"gan_training_log_{run_timestamp}.log"
    logger = setup_logging(os.path.join('logs', log_file_name))

    config = GANConfig('configs/config.yaml')
    
    train_and_save_generator(model_save_path, logger, config, run_timestamp)
