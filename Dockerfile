# Use a Python 3.12 base image (Bookworm is Debian 12, compatible with 3.12)
FROM python:3.12-slim-bookworm

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (including git-lfs and CUDA-related libs for torch)
# Order matters: apt-get update before apt-get install
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    cmake \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --skip-smudge # Initialize Git LFS inside the container

# Copy requirements.txt first to leverage Docker layer caching
COPY requirements.txt ./

# Install Python dependencies from requirements.txt
# Using 'pip' is fine as it usually points to the correct Python version's pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
# This copies everything from your local repo root into /app in the container
COPY . /app

# Ensure correct PYTHONPATH for your src directory
# This tells Python to look inside '/app/src' for modules when importing
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Expose the port Streamlit runs on
EXPOSE 8501

# Healthcheck for monitoring container status
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=5 CMD curl --fail http://localhost:8501/_stcore/health || exit 1
# Command to run your Streamlit app
# Ensure the port matches EXPOSE and HEALTHCHECK
ENTRYPOINT ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]