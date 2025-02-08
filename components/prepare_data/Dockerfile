# Use an official lightweight Python runtime as a parent image.
FROM python:3.12-slim

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . .
