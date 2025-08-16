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
import yaml # New: Import for reading config

# --- 1. Model Architectures ---
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

# --- 2. FastAPI Application Setup ---
app = FastAPI(title="MNIST GAN Generator")
templates = Jinja2Templates(directory="app/templates")

# Global variables for the models and config
generator = None
classifier = None
latent_dim = 100
img_shape = (1, 28, 28)
device = "cuda" if torch.cuda.is_available() else "cpu"
# New: Global variable for num predictions
NUM_PREDICTIONS = 3

# Function to load the models and config
def load_models():
    global generator
    global classifier
    global NUM_PREDICTIONS

    # New: Load the configuration file
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    NUM_PREDICTIONS = config.get('num_predictions_to_show', 3)
    
    # Load Generator model
    gen_model_path = "model/generator_model.pth"
    if os.path.exists(gen_model_path):
        generator = Generator(latent_dim, img_shape).to(device)
        generator.load_state_dict(torch.load(gen_model_path, map_location=device))
        generator.eval()
        print("Generator model loaded successfully.")
    else:
        print("Warning: Generator model file not found.")

    # Load Classifier model
    cls_model_path = "model/classifier_model.pth"
    if os.path.exists(cls_model_path):
        classifier = Classifier().to(device)
        classifier.load_state_dict(torch.load(cls_model_path, map_location=device))
        classifier.eval()
        print("Classifier model loaded successfully.")
    else:
        print("Warning: Classifier model file not found.")

# Load models on application startup
load_models()

# --- 3. API Routes ---
@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/generate_image")
async def generate_image():
    if generator is None or classifier is None:
        return {"error": "One or more models not loaded. Please run model preparation scripts."}
    
    with torch.no_grad():
        # Generate image with GAN
        z = torch.randn(1, latent_dim, device=device)
        generated_img_tensor = generator(z).detach().cpu()
        
        # Classify the generated image with the classifier
        classifier_input = generated_img_tensor.to(device)
        output = classifier(classifier_input)
        
        # New: Get the top-k predictions
        probabilities = F.softmax(output, dim=1)
        top_k_probs, top_k_preds = torch.topk(probabilities, NUM_PREDICTIONS, dim=1)
        
        predictions_list = []
        for i in range(NUM_PREDICTIONS):
            predictions_list.append({
                "prediction": top_k_preds[0][i].item(),
                "confidence": f"{top_k_probs[0][i].item() * 100:.2f}%"
            })
        
        # Convert image tensor to Base64 string for the UI
        img_array = (generated_img_tensor[0] * 0.5 + 0.5) * 255
        img_array = img_array.to(torch.uint8).numpy()
        img_array = img_array.reshape(28, 28)
        img = Image.fromarray(img_array, 'L')
        
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
    return {
        "image": f"data:image/png;base64,{img_base64}",
        "predictions": predictions_list # New: Return the list of top predictions
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
