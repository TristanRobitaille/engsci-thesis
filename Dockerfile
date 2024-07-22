FROM ubuntu:24.10

# Set environment variables to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    software-properties-common \
    && apt-get clean

# Python
RUN apt-get install -y python3 python3-pip

#----- LaTeX dependencies -----#
RUN apt-get install --no-install-recommends -y \
    biber \
    latexmk \
    texlive-full
RUN apt-get install -y python3-scienceplots

#----- ASIC functional simulation dependencies -----#
ENV N_STO_INT_RES=20
ENV N_STO_PARAMS=20
RUN apt-get install -y \
    libboost-all-dev \
    libhdf5-dev \
    libarmadillo-dev

# Clean up
RUN rm -rf /var/lib/apt/lists/* /usr/local/src/*

WORKDIR /tmp


# TODO
# -Test fixed-point accuracy study
# -Python dependencies for TensorFlow
# -Python dependencies for CocoTB
# -Run it with VSCode
# -Avoid using texlive-full and install only the necessary packages (very long build time and large image size)