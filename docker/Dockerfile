FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
WORKDIR /workspace

# python and basic dependencies for portfolio experiments
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    bash-completion \
    build-essential \
    curl \
    git \
    vim \
    htop \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements/requirements.txt requirements.txt
ENV PIP_DEFAULT_TIMEOUT=600
RUN pip install --no-cache-dir --retries 5 --timeout 600 -r requirements.txt