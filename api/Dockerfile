FROM python:3.11-slim

WORKDIR /usr/src/api

COPY . /usr/src/api

RUN pip install --no-cache-dir -r /usr/src/api/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--reload", "--host=0.0.0.0", "--port=8000"]