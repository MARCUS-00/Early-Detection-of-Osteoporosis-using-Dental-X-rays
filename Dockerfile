# ---- OsteoScan production image ----
FROM python:3.11-slim

WORKDIR /app

# System libs required by OpenCV (kept for safety even with the headless build).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application.
COPY . .

# Writable data directory (uploads, reports, SQLite DB live here via DATA_DIR).
# chmod 777 so the non-root user on Hugging Face Spaces can write to it.
RUN mkdir -p /app/data && chmod -R 777 /app/data

EXPOSE 5000
ENV FLASK_ENV=production

# Shell form so ${PORT} expands at runtime. One worker keeps RAM within
# constrained tiers; 120s timeout covers cold-start model loading.
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 120 wsgi:app
