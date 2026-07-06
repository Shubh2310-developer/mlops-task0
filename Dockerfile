FROM python:3.9-slim

WORKDIR /app

# Copy and install dependencies first (leverage layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application script and default assets
COPY run.py .
COPY config.yaml .
COPY data.csv .

# Set default execution command
CMD ["python", "run.py", "--input", "data.csv", "--config", "config.yaml", "--output", "metrics.json", "--log-file", "run.log"]
