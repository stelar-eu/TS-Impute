FROM ubuntu:bionic

# LABEL about the custom image
LABEL maintainer="pbetchavas@hotmail.gr"
LABEL version="0.1"
LABEL description="This is custom Docker Image for \
imputation algorithms."

# WORKDIR
WORKDIR /app
COPY AlgoCollection/ /app/AlgoCollection/
COPY stelarImputation/ /app/stelarImputation/
COPY setup.py /app/setup.py
COPY main.py /app/main.py
COPY run.sh /app/run.sh
 
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
RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
apt-get update && \
apt-get install cmake -y

RUN apt-get install -y kitware-archive-keyring

# Install Armadillo
RUN wget http://sourceforge.net/projects/arma/files/armadillo-10.8.2.tar.xz
RUN tar -xvf armadillo-10.8.2.tar.xz
RUN cd armadillo-10.8.2 && \
./configure && \
make && \
make install && \
cd ..

# Install pybind11 + other libraries
RUN pip install --upgrade pip && \
	pip install --no-cache-dir pybind11==2.9.2 


# Install Custom library
RUN cd /app/AlgoCollection && \
mkdir build && \
cd build && \
cmake .. -DPYTHON_LIBRARY_DIR=/venv/lib/python3.8/site-packages -DPYTHON_EXECUTABLE=/venv/bin/python3 -Dpybind11_DIR=/venv/lib/python3.8/site-packages/pybind11/share/cmake/pybind11 && \
make && \
make install && \
cd .. && \
cd ..

# Install stelarImputation library
RUN pip3 install --no-cache-dir torch==2.1.2+cu121  -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir torch-cluster==1.6.3 torch-sparse==0.6.18 torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
RUN cd /app && pip install --no-cache-dir .

RUN chmod +x run.sh
ENTRYPOINT ["./run.sh"]