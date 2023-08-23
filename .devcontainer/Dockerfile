FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

RUN apt-get update \
    && apt-get install python3 python3-pip -y

COPY .devcontainer/requirements-pinned.txt /tmp

RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements-pinned.txt \
    && rm /tmp/requirements-pinned.txt

RUN pip install \
    ipywidgets \
    ipython
