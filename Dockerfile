FROM ubuntu:24.10

# Set environment variables to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for the ASIC functional simulation
ENV N_STO_INT_RES=20
ENV N_STO_PARAMS=20

# ASIC functional simulation dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && apt-get clean

RUN apt-get install -y \
    libboost-all-dev \
    libhdf5-dev \
    libarmadillo-dev

WORKDIR /tmp