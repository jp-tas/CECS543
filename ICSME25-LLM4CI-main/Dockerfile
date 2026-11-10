# Use Python 3.13 base
FROM python:3.13-slim

# Install system dependencies including build tools for packages with C extensions
RUN apt-get update && apt-get install -y \
    build-essential libffi-dev libssl-dev \
    wget curl unzip \
    python3-dev chromium chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /project

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Make chromedriver available in the current working directory
RUN ln -s /usr/bin/chromedriver ./chromedriver

# Install Ollama CLI
RUN curl -s https://ollama.com/install.sh | bash

# Set environment variable for OpenAI API Key
ENV OPENAI_API_KEY=""

# Default command to run when the container starts
CMD ollama start & bash
