FROM python:3.11-slim

WORKDIR /app

# psycopg2-binary needs no build tools; docker SDK + mlflow + prometheus-client
# are pure Python + C extensions already bundled in wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scheduler.py force_tick.py ./

# /metrics exposed here; prometheus scrape target in ops/prometheus/prometheus.yml.
EXPOSE 8000

CMD ["python", "scheduler.py"]
