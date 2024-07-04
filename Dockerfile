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
  && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory
COPY . .

# Expose the port the app runs on
EXPOSE 8042

# Command to run the application
CMD ["python", "app.py"]
