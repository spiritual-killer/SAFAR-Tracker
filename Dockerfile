# Step 1: Create the Dockerfile
nano Dockerfile

# Add the following content to the Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies including distutils
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3-distutils

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Check if Streamlit is installed
RUN pip show streamlit || { echo "Streamlit is not installed"; exit 1; }

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "Streamlit.py"]

# Save and close the file
