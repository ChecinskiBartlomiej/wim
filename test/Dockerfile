# Dockerfile for MLP training with PyTorch
FROM python:3.12-slim

# Set working directory
WORKDIR /workspace

# Install PyTorch with CUDA 12.6 support (matching WIM cluster)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu126 \
    numpy

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python"]
