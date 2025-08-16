import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

# Create a TestClient instance for your FastAPI app
client = TestClient(app)

def test_read_main_returns_html():
    """
    Tests that the root endpoint '/' returns a successful response.
    This is a basic check to ensure the server is running.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "MNIST GAN Generator" in response.text

def test_generate_image_returns_valid_json():
    """
    Tests that the '/generate_image' endpoint returns a valid JSON response
    with a Base64 encoded image string.
    """
    # The prepare_model.py and prepare_classifier.py scripts
    # need to be run first to generate the model files.
    # We will skip this test if the model files are not present.
    if not os.path.exists("model/generator_model.pth") or not os.path.exists("model/classifier_model.pth"):
        pytest.skip("Model files not found, skipping API test.")
    
    response = client.get("/generate_image")
    assert response.status_code == 200
    data = response.json()
    assert "image" in data
    assert "prediction" in data
    assert "confidence" in data
    assert data["image"].startswith("data:image/png;base64,")

