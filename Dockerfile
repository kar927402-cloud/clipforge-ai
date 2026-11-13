FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV WHISPER_MODEL=base
ENV MAX_UPLOAD_MB=500
CMD ["sh", "-c", "gunicorn -w 1 -k gthread --threads 2 -b 0.0.0.0:${PORT:-8000} app:app"]
