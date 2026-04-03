# Step 1: Base image
FROM python:3.11-slim

# Step 2: Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Step 3: Set work directory
WORKDIR /app

# Step 4: Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libportaudio2 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Step 5: Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy project files
COPY . .

# Step 7: Expose Flask port
EXPOSE 5000

# Step 8: Run the app
CMD ["python", "app.py"]