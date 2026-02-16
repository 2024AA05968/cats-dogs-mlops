# Use Python 3.12 to match your current environment
FROM python:3.12-slim

# Prevent Python from writing .pyc and buffer logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# (Optional but helpful) system deps for torch CPU / general runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better layer caching)
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code and model artifact
COPY app /app/app
COPY models /app/models

# Expose API port
EXPOSE 8000

# IMPORTANT: Use python -m uvicorn (avoids blocked uvicorn.exe scenarios)
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]