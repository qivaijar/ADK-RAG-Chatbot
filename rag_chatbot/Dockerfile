# Use a lightweight Python base
FROM python:3.12-slim

WORKDIR /app

# Copy project files
COPY . rag_chatbot/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r rag_chatbot/requirements.txt --no-cache-dir

# Expose port Cloud Run expects
EXPOSE 8080

# Set Python to unbuffered mode for logging
ENV PYTHONUNBUFFERED=1

# Run your ADK API server
CMD ["adk", "api_server", "."]

