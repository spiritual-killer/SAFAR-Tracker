# Step 1: Create the Dockerfile
nano Dockerfile

# Add the following content to the Dockerfile
# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6

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

# Step 2: Update requirements.txt
# Ensure your requirements.txt file lists only Python dependencies:
streamlit==1.24.0
opencv-python-headless==4.10.0.84
numpy
pandas
ultralytics==8.3.19
scikit-learn
scipy

# Step 3: Add, Commit, and Push to GitHub
# Add the Dockerfile and requirements.txt to the staging area
git add Dockerfile requirements.txt

# Commit the changes with a message
git commit -m "Add Dockerfile and update requirements.txt"

# Push the changes to the main branch on GitHub
git push origin main
