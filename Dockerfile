# Step 1: Base image
FROM python:3.10.11-slim

# Step 2: Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Step 3: Set work directory
WORKDIR /app

FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libportaudio2 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# HuggingFace requires port 7860
EXPOSE 7860

# Run with gunicorn (BEST PRACTICE)
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]