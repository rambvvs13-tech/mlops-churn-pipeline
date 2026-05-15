FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/inference/app.py .
COPY models/best_churn_model.pkl .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
docker build -t churn-api .
docker run -p 8000:8000 churn-api