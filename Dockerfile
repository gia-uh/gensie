# Use a slim Python image
FROM python:3.13-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app/gensie-lib

# 1. Copy only the dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock README.md ./

# 2. Install dependencies (excludes the project itself)
# This layer will stay cached unless pyproject.toml or uv.lock changes.
RUN uv sync --frozen --no-install-project --no-dev

# 3. Copy the actual source code
COPY . .

# 4. Install the project in editable mode
# This allows mounting a volume over /app/gensie-lib to see changes without rebuilding.
RUN uv pip install -e .

# Expose the FastAPI port
EXPOSE 8000

# Ensure the virtual environment's binaries are in the PATH
ENV PATH="/app/gensie-lib/.venv/bin:$PATH"

# Set default environment variables for the agent
ENV PARTICIPANT_PATH="gensie.baseline.OfficialParticipant"
ENV AGENT_MODEL="gpt-4o-mini"
ENV OPENAI_BASE_URL=""
ENV OPENAI_API_KEY="sk-dummy"

# Run the server via the CLI
# USAGE: docker run -p 8000:8000 -v $(pwd):/app/gensie-lib gensie-baseline
ENTRYPOINT ["gensie", "serve"]
