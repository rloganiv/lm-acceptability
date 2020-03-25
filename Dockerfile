FROM python:3.7.7-stretch

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PATH /usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Tell nvidia-docker the driver spec that we need as well as to
# use all available devices, which are mounted at /usr/local/nvidia.
# The LABEL supports an older version of nvidia-docker, the env
# variables a newer one.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

WORKDIR /stage/lm-acceptability

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy select files needed for installing requirements.
# We only copy what we need here so small changes to the repository does not trigger re-installation of the requirements.
COPY setup.py .
COPY README.md .

COPY acceptability/ acceptability/
COPY pytest.ini pytest.ini
COPY configs/ configs/
COPY setup.py setup.py
COPY README.md README.md

RUN pip install --editable .

# TODO: Maybe this...
# Caching models when building the image makes a dockerized server start up faster, but is slow for
# running tests and things, so we skip it by default.
#  ARG CACHE_MODELS=false
#  RUN ./scripts/cache_models.py

EXPOSE 8000
CMD ["/bin/bash"]