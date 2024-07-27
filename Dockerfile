FROM ubuntu:23.10
LABEL description="EngSci Thesis Docker Image"

# Set environment variables to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

#----- General dependencies -----#
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    software-properties-common \
    pkg-config \
    gdb \
    locales \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_CA.UTF-8

#----- VSCode server -----#
RUN mkdir -p /root/.vscode-server/extensions
RUN mkdir -p /root/.vscode-server-insiders/extensions

# Install boost manually since the required version is not available on Ubuntu 23.10
RUN apt-get update && apt-get install -y wget && \
    wget 'http://downloads.sourceforge.net/project/boost/boost/1.84.0/boost_1_84_0.tar.gz' -c && \
    tar -zxvf boost_1_84_0.tar.gz && \
    cd boost_1_84_0 && \
    ./bootstrap.sh && \
    ./b2 --with=all -j8 install && \
    cd ../ && rm boost_1_84_0.tar.gz && rm -r boost_1_84_0

#----- Python -----#
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv $VIRTUAL_ENV

#----- LaTeX dependencies -----#
RUN apt-get update && apt-get install --no-install-recommends -y \
    biber \
    latexmk \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-science \
    texlive-bibtex-extra \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install scienceplots

#----- ASIC functional simulation dependencies -----#
ENV N_STO_INT_RES=20
ENV N_STO_PARAMS=20
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libarmadillo-dev \
    && rm -rf /var/lib/apt/lists/*

#----- ASIC RTL dependencies -----#
RUN apt-get update && apt-get install -y \
    verilator && \
    rm -rf /var/lib/apt/lists/*
COPY asic/rtl/requirements.txt .
RUN pip3 install --upgrade -r requirements.txt

#----- Tensorflow dependencies -----#
COPY python_prototype/requirements.txt .
RUN pip3 install --upgrade -r requirements.txt

# Clean up
WORKDIR /tmp