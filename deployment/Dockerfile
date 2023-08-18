FROM python:3.9-slim AS build

RUN apt-get update

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY deployment/requirements.txt .
RUN pip install -r requirements.txt --use-pep517

FROM python:3.9-slim

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY . /mlops_pipeline
WORKDIR /mlops_pipeline