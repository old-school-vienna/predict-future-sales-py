FROM python:3.8.6-slim-buster

RUN python -m pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt

