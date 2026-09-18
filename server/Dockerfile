# Base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# System dependencies for OpenCV and other C-based python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy server application code
COPY . /app/server

# Set Python path to ensure module resolution works
ENV PYTHONPATH="/app"

# Command to run the RunPod serverless handler
CMD ["python", "-u", "/app/server/handler.py"]
