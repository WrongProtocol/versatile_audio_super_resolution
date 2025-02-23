# Use the official Python image from the Docker Hub
FROM python:3.8-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends git procps ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir cog pyloudnorm

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]

# to run it, do:
#   docker build -t audio-super-resolution .
#   docker run -d -p 8002:8000 --gpus=all --name AudioSuperRes audio-super-resolution
#
