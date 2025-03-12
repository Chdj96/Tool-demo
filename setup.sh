#!/bin/bash

# Update package list and install required system dependencies (if needed)
sudo apt-get update && sudo apt-get install -y python3-pip

# Upgrade pip
pip install --upgrade pip

# Install dependencies from requirements file
pip install -r requirements.txt

# Run Streamlit to verify installation
streamlit run Homepage.py

