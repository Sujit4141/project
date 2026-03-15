# Use a slim Python image as base
FROM python:3.9-slim

# Install system dependencies needed by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Flask will run on
EXPOSE 5000

# Use gunicorn to serve the app (production)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]