FROM python:3.13-slim

WORKDIR /app

# Install system dependencies for sentence-transformers and numpy
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire gensie package
COPY . /app/gensie-lib

# Install the package in editable mode or normally
RUN pip install --no-cache-dir "/app/gensie-lib"

# Expose the FastAPI port
EXPOSE 8000

# Set default environment variables
ENV AGENT_PATH="gensie.baseline.BasicAgent"
ENV AGENT_MODEL="gpt-4o-mini"
ENV OPENAI_BASE_URL=""
ENV OPENAI_API_KEY="sk-dummy"

# Run the server via the CLI
ENTRYPOINT ["gensie", "serve"]
