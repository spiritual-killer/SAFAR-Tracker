#!/bin/bash

# Install system dependencies
apt-get update
apt-get install -y libgl1-mesa-glx

# Install Python dependencies
pip install -r requirements.txt
pip install -r packages.txt
