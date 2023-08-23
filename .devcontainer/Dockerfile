FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

COPY .devcontainer/requirements-pinned.txt /tmp

RUN pip install --upgrade pip \
    && pip install \
    -r /tmp/requirements-pinned.txt \
    ipykernel \
    ipython \
    ipywidgets \
    && rm /tmp/requirements-pinned.txt
