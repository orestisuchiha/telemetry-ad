FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command can be overridden in docker run
CMD ["python", "scripts/train_offline.py", "--dataset", "nab", "--series", "ec2_cpu_utilization_5f5533"]
