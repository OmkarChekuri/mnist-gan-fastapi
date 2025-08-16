import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time
import logging

# --- Logging Setup ---
def setup_logging(log_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# --- Model Architecture ---
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_and_save_classifier(output_path, logger):
    logger.info("--- Starting Classifier Training and Export ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 5
    batch_size = 64
    
    os.makedirs('data/mnist', exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_train_dataset = datasets.MNIST(root='data/mnist', train=True, download=True, transform=transform)
    train_loader = DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True)
    
    model = Classifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                logger.info(f"Classifier Train Epoch: {epoch+1}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)] Loss: {loss.item():.6f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    logger.info(f"\nClassifier model successfully saved to {output_path}")
    logger.info("--- Classifier Export Complete ---")


if __name__ == '__main__':
    model_save_path = os.path.join('model', 'classifier_model.pth')
    log_file_name = f"classifier_training_log_{int(time.time())}.log"
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger = setup_logging(os.path.join('logs', log_file_name))
    train_and_save_classifier(model_save_path, logger)
