FROM python:3.11-slim

WORKDIR /usr/src/app

COPY . /usr/src/app

RUN pip install --no-cache-dir -r /usr/src/app/requirements.txt

EXPOSE 8500

CMD ["streamlit", "run", "/usr/src/app/app.py", "--server.port", "8500"]