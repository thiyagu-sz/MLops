
FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN python train.py

CMD exec gunicorn --bind :$PORT app:app
