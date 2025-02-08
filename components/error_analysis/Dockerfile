# Use an official lightweight Python runtime as a parent image.
FROM python:3.12-slim

# Copy the requirements file to leverage Docker cache during builds
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .
