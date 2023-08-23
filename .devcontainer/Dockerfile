FROM ubuntu:latest

RUN apt-get update \
    && apt-get install python3 python3-pip -y

# COPY .devcontainer/requirements-pinned.txt /tmp
COPY requirements-pinned.txt /tmp

# RUN pip install --upgrade pip \
#     && wget https://download.pytorch.org/whl/torch-2.0.1-cp311-cp311-manylinux2014_aarch64.whl \
#     && pip install torch-2.0.1-cp311-cp311-manylinux2014_aarch64.whl \
#     && pip install -r /tmp/requirements-pinned.txt \
#     && rm /tmp/requirements-pinned.txt \
#     && rm torch-2.0.1-cp311-cp311-manylinux2014_aarch64.whl

RUN pip install --upgrade pip \
    && pip install -r /tmp/requirements-pinned.txt \
    && rm /tmp/requirements-pinned.txt
