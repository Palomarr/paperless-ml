FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir torch mlflow pyyaml

CMD ["python", "train.py"]