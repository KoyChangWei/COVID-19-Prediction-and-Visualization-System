#!/bin/bash

# Build script for Render deployment
echo "🚀 Starting COVID-19 Prediction System build..."

# Check Python version
echo "📍 Current Python version:"
python --version

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Build completed successfully!" 