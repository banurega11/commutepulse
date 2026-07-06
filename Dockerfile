FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate the dataset at build time so it's baked into the image
RUN python data_gen.py --rows 200000 --out commute_data.csv

# Cloud Run injects $PORT; gunicorn binds to it
ENV PORT=8080
EXPOSE 8080

CMD exec gunicorn --bind 0.0.0.0:${PORT} --workers 2 --timeout 60 app:app
