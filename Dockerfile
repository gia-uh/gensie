# Use a slim Python image
FROM python:3.13-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# 1. Copy only the dependency files first
COPY pyproject.toml uv.lock README.md ./

# 2. Install external dependencies only
# We use --no-install-project if we were using uv sync, 
# but with uv pip we can just install the requirements.
# To avoid building the project, we use uv pip install on the dependencies.
RUN uv pip install --system -r pyproject.toml

# 3. Copy the actual source code into a subfolder
COPY . /app/gensie-lib

# 4. Install the project in editable mode
RUN uv pip install --system -e /app/gensie-lib

# Expose the FastAPI port
EXPOSE 8000

# Set default environment variables for the agent
ENV PARTICIPANT_PATH="gensie.baseline.OfficialParticipant"
ENV OPENAI_BASE_URL=""
ENV OPENAI_API_KEY="sk-dummy"

# Run the server via the CLI
ENTRYPOINT ["gensie"]
