import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import logging
import time

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

# --- 2. Model Architecture ---
# This is a simple Generator from the GAN notebook.
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

# --- 3. Training Script for Generator ---
def train_and_save_generator(output_path, logger, num_samples=1000):
    """
    Trains a small GAN Generator on dummy data and saves the model state.
    """
    logger.info("--- Starting Model Training and Export ---")
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = 100
    img_shape = (1, 28, 28)
    lr = 0.0002
    epochs = 10
    batch_size = 64

    # Create dummy data for a quick training run
    dummy_noise = torch.randn(num_samples, latent_dim)
    dummy_dataset = TensorDataset(dummy_noise)
    dataloader = DataLoader(dummy_dataset, batch_size=batch_size)

    # Initialize Generator and Optimizer
    generator = Generator(latent_dim, img_shape).to(device)
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # Dummy Training Loop
    generator.train()
    epoch_metrics = []
    for epoch in range(epochs):
        for i, noise in enumerate(dataloader):
            noise = noise[0].to(device)
            valid = torch.ones(noise.size(0), 1).to(device)
            
            optimizer_g.zero_grad()
            gen_imgs = generator(noise)
            g_loss = criterion(torch.sigmoid(gen_imgs), valid)
            g_loss.backward()
            optimizer_g.step()

        logger.info(f"Epoch {epoch+1}/{epochs}, Generator Loss: {g_loss.item():.4f}")
        epoch_metrics.append({'epoch': epoch + 1, 'loss': g_loss.item()})

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the trained model
    torch.save(generator.state_dict(), output_path)
    logger.info(f"\nGenerator model successfully saved to {output_path}")
    logger.info(f"--- Epoch Metrics Log ---")
    logger.info(str(epoch_metrics))
    logger.info("--- Model Export Complete ---")


if __name__ == '__main__':
    # Define the path to save the model and the log file
    model_save_path = os.path.join('model', 'generator_model.pth')
    log_file_name = f"gan_training_log_{int(time.time())}.log"
    
    # Setup logging
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger = setup_logging(os.path.join('logs', log_file_name))
    
    # Train and save the model
    train_and_save_generator(model_save_path, logger)
