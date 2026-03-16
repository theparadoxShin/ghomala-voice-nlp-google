FROM python:3.11-slim

WORKDIR /app

# ffmpeg for audio format conversion (M4A→PCM for Gemini Live API)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY nam_sa_agent/ nam_sa_agent/
COPY data/ data/

# Cloud Run sets PORT env var
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
