FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model from HF at build time (avoids LFS pointer issues)
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Santoshp123/flower-species-classifier', filename='my_flower_cnn.h5', local_dir='.')"

# Copy app files
COPY . .

# Expose Streamlit port
EXPOSE 7860

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
