import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import io
from PIL import Image
import base64

# --- 1. Model Architecture (Must match the architecture used in training) ---
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

# --- 2. FastAPI Application Setup ---
app = FastAPI(title="MNIST GAN Generator")
templates = Jinja2Templates(directory="app/templates")

# Global variables for the model
generator = None
latent_dim = 100
img_shape = (1, 28, 28)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to load the model (will be run on application startup)
def load_model():
    global generator
    model_path = "model/generator_model.pth"
    if os.path.exists(model_path):
        generator = Generator(latent_dim, img_shape).to(device)
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.eval()
        print("Generator model loaded successfully.")
    else:
        print("Warning: Model file not found. Please run prepare_model.py first.")
        generator = Generator(latent_dim, img_shape).to(device)
        generator.eval()

# Load the model on startup
load_model()

# --- 3. API Routes ---

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """Serve the HTML front-end for the application."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/generate_image")
async def generate_image():
    """
    Generate a new image using the GAN Generator and return it as a Base64 string.
    """
    if generator is None:
        return {"error": "Model not loaded."}
        
    with torch.no_grad():
        # Generate a random noise vector
        z = torch.randn(1, latent_dim, device=device)
        
        # Pass the noise through the Generator to create an image
        generated_img = generator(z).detach().cpu().numpy()
        
        # Denormalize the image from [-1, 1] to [0, 255] and convert to uint8
        img_array = (generated_img[0] * 0.5 + 0.5) * 255
        img_array = img_array.astype(np.uint8)
        
        # Reshape the image array to a 28x28 grayscale image
        img_array = img_array.reshape(28, 28)
        
        # Create a PIL Image object from the numpy array
        img = Image.fromarray(img_array, 'L')  # 'L' mode for grayscale
        
        # Save the image to an in-memory buffer
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        
        # Encode the image data to a Base64 string
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
    return {"image": f"data:image/png;base64,{img_base64}"}

# --- 4. Main execution for Uvicorn server ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
