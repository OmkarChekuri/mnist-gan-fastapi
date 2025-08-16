# Use a Python image that includes the necessary libraries for PyTorch and FastAPI.
# This ensures that our environment is consistent and reproducible.
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
# `gunicorn` is a production-ready web server that works well with FastAPI.
# `uvicorn` is included to run the application.
RUN pip install --no-cache-dir gunicorn uvicorn[standard]
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the FastAPI application will be listening on
EXPOSE 8000

# Set the entrypoint to run the FastAPI application using Gunicorn
# Gunicorn is a production-ready WSGI server that can be used with Uvicorn workers.
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "app.main:app"]
