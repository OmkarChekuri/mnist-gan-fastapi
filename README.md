# MNIST GAN with FastAPI and HTML UI

This project implements a Generative Adversarial Network (GAN) to generate handwritten digits, which are then served through a web application built with FastAPI and a custom HTML front-end. The project is designed for reproducibility and demonstrates a complete MLOps workflow, from model training to application deployment.

## Project Overview

![GAN Application Screenshot](uploaded:application.PNG-14e61ad1-7aa6-401d-9aef-e1a19d86446c)

The core of this project consists of:

* A **Generator model** trained on the MNIST dataset to create new handwritten digits.

* A **FastAPI server** that exposes a `/generate_image` endpoint to serve these new images.

* A simple **HTML front-end** that allows users to interact with the model and view the generated digits in real-time.

The project is structured to separate the model training pipeline from the web application, making it easy to manage and deploy.

## Folder Structure

```
.
├── app/                     # Main application code
│   ├── main.py              # FastAPI endpoint to generate images
│   └── templates/           # HTML templates for the UI
│       └── index.html
├── model/                   # Trained model files
│   └── generator_model.pth  # The saved Generator model
├── logs/                    # Training logs and other run artifacts
├── tests/                   # Application test cases
│   └── test_app.py
├── .github/                 # GitHub Actions CI pipeline
│   └── workflows/
│       └── ci.yml
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── prepare_model.py         # Script to train and save the GAN model

```

## Setup and Installation

### Prerequisites

* Python 3.8+

* A CUDA-enabled GPU (optional, but recommended)

### Local Environment Setup

1.  **Clone the repository and navigate into it:**

    ```
    git clone [https://github.com/your_username/mnist-gan-fastapi.git](https://github.com/your_username/mnist-gan-fastapi.git)
    cd mnist-gan-fastapi

    ```

2.  **Activate your virtual environment:**

    ```
    source venv/bin/activate

    ```

3.  **Install dependencies:**

    ```
    pip install -r requirements.txt

    ```

4.  **Prepare the model:** Run the training script to generate and save your `generator_model.pth` file.

    ```
    python prepare_model.py

    ```

## Running the Application

To run the FastAPI application, use the `uvicorn` server.

```
python -m uvicorn app.main:app --reload

```

This will start the server, and you can view the application in your browser at `http://127.0.0.1:8000`.

## CI/CD Pipeline

A GitHub Actions pipeline is configured to run tests on the FastAPI application every time code is pushed. This ensures the application remains functional and correctly configured. The pipeline will:

* Install dependencies.

* Run tests on the FastAPI endpoints.

This continuous integration process ensures that your application is always ready for deployment