FROM ubuntu:focal

# WORKDIR
WORKDIR /app
COPY . /app/

RUN : \
    && apt-get update \
	&& apt-get install -y curl jq \
	&& DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
		python3-distutils \
	&& DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
		python3.8-dev \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :

RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

WORKDIR /app/utils/stelarImputation

# Update Ubuntu Software repository + Essenstials
RUN apt update && \
apt-get install build-essential -y && \
apt-get install libopenmpi-dev -y && \
apt-get install libopenblas-dev -y && \
apt-get install liblapack-dev -y && \
apt-get install libmlpack-dev -y 

# Install cmake
RUN apt-get update && apt-get install -y ca-certificates gpg wget

RUN test -f /usr/share/doc/kitware-archive-keyring/copyright || \
    wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
    gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
apt-get update && \
apt-get install cmake -y

RUN apt-get install -y kitware-archive-keyring git

# Install carma
RUN git clone https://github.com/RUrlus/carma.git \
&& cd carma \
&& git checkout v0.8.0 \
&& mkdir build && cd build \
&& cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DCARMA_INSTALL_LIB=ON -DCMAKE_BUILD_TYPE=Release \
&& make \
&& make install \
&& ln -sf /usr/local/share/carma/cmake/carmaConfig.cmake /usr/local/share/carma/cmake/CarmaConfig.cmake \
&& ln -sf /usr/local/share/carma/cmake/carmaConfigVersion.cmake /usr/local/share/carma/cmake/CarmaConfigVersion.cmake

# Install armadillo
RUN ARMA_VER=10.8.2 \
&& wget https://downloads.sourceforge.net/project/arma/armadillo-${ARMA_VER}.tar.xz \
&& tar xf armadillo-${ARMA_VER}.tar.xz \
&& cd armadillo-${ARMA_VER} \
&& ./configure -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
&& make \
&& make install 

# Install pybind11 + other libraries
RUN pip install --upgrade pip setuptools wheel && \
pip install --no-cache-dir numpy>=1.14 pybind11>=2.12.0


# Install Custom library
WORKDIR /app/utils/stelarImputation/AlgoCollection 
RUN mkdir build && \
cd build && \
cmake .. -DPYTHON_LIBRARY_DIR=/venv/lib/python3.8/site-packages -DPYTHON_EXECUTABLE=/venv/bin/python3 -Dpybind11_DIR=/venv/lib/python3.8/site-packages/pybind11/share/cmake/pybind11 && \
make && \
make install

# Install stelarImputation library
WORKDIR /app/utils/stelarImputation
RUN pip install --no-cache-dir .

WORKDIR /app/utils/LLMs
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "from pypots.imputation import SAITS"

RUN chmod +x run.sh
ENTRYPOINT ["./run.sh"]
