# Lightweight Python runtime
FROM python:3.12-slim

WORKDIR /app

# Copy everything into container
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

# Expose port
EXPOSE 8000

# Set Gradio to listen on all interfaces
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=8000
ENV PYTHONUNBUFFERED=1

# Run Gradio app
CMD ["python", "app.py"]

