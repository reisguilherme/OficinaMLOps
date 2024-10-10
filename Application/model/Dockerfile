# Use an official PyTorch image with CUDA support
FROM pytorch/pytorch

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt --no-cache-dir 

# Copy the rest of the application code
COPY . .

# Expose the port that the application will run on
EXPOSE 8000

# Set environment variables for CUDA (optional but recommended)
ENV CUDA_VISIBLE_DEVICES=all

# Command to run the application (endpoint.py)
CMD ["python3", "endpoint.py"]
