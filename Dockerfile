# Multi-stage Dockerfile for Titanic ML Project
# Stage 1: Base image with dependencies
FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Development image
FROM base as development

# Copy all project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models reports

# Download dataset
RUN python scripts/download_data.py

# Default command for development
CMD ["bash"]

# Stage 3: Training image
FROM development as train

# Train the model
CMD ["python", "src/train.py"]

# Stage 4: Production image (lightweight)
FROM python:3.10-slim as production

WORKDIR /app

# Install only runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only necessary files
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY --from=train /app/models/ ./models/
COPY --from=train /app/data/ ./data/

# Create a non-root user
RUN useradd -m -u 1000 mluser && \
    chown -R mluser:mluser /app

USER mluser

# Expose any necessary ports (if needed for API in future)
# EXPOSE 8000

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "src/evaluate.py", "--model", "models/titanic_model_random_forest.pkl", "--use-test"]
