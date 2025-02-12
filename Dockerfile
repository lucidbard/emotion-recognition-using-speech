# Use the official Python base image
FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
  gcc \
  g++ \
  pkg-config \
  libfreetype6-dev \
  libpng-dev \
  portaudio19-dev \
  libsndfile1 \
  && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install specific version of numba compatible with librosa 0.6.3
RUN pip install --no-cache-dir numba==0.48

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Expose the port the app runs on
EXPOSE 8042

# Command to run the application
CMD ["python", "app.py"]