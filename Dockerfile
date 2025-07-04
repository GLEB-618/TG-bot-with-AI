FROM python:3.10-alpine AS build

WORKDIR /app

COPY requirements.txt ./

RUN pip install requirements.txt

COPY . ./

ARG PYTHONPATH=src

FROM build AS main

ENTRYPOINT [ "python", "src\bot\main.py" ]