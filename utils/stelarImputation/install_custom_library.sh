sudo apt-get update \
&& sudo apt-get install -y build-essential libopenmpi-dev libopenblas-dev liblapack-dev libmlpack-dev \
&& sudo apt-get install -y python3.8-dev \
&& sudo apt-get update \
&& sudo apt-get install -y ca-certificates gpg wget \
&& wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
&& UBUNTUCODENAME=$(lsb_release -cs) \
&& echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/' $UBUNTUCODENAME 'main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null \
&& sudo apt-get update \
&& sudo apt-get install -y cmake \
&& sudo rm -rf carma \
&& git clone https://github.com/RUrlus/carma.git \
&& cd carma \
&& git checkout v0.8.0 \
&& mkdir build && cd build \
&& cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DCARMA_INSTALL_LIB=ON -DCMAKE_BUILD_TYPE=Release \
&& make -j$(nproc) \
&& sudo make install \
&& cd ../../ \
&& sudo ln -sf /usr/local/share/carma/cmake/carmaConfig.cmake /usr/local/share/carma/cmake/CarmaConfig.cmake \
&& sudo ln -sf /usr/local/share/carma/cmake/carmaConfigVersion.cmake /usr/local/share/carma/cmake/CarmaConfigVersion.cmake \
&& ARMA_VER=10.8.2 \
&& sudo rm -rf armadillo-${ARMA_VER}.tar.xz \
&& wget https://downloads.sourceforge.net/project/arma/armadillo-${ARMA_VER}.tar.xz \
&& sudo rm -rf armadillo-${ARMA_VER} \
&& tar xf armadillo-${ARMA_VER}.tar.xz \
&& cd armadillo-${ARMA_VER} \
&& ./configure -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
&& make \
&& sudo make install \
&& cd ../../ \
&& pip install --upgrade pip setuptools wheel \
&& pip install numpy>=1.14 pybind11>=2.12.0 \
&& PYTHONENVLIB=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
&& PYTHONENVEXE=$(which python3) \
&& cd AlgoCollection \
&& sudo rm -rf build \
&& mkdir build && cd build \
&& cmake .. -DPYTHON_LIBRARY_DIR=$PYTHONENVLIB -Dpybind11_DIR=$PYTHONENVLIB"/pybind11/share/cmake/pybind11" \
&& make \
&& make install \
&& cd ../.. \
&& pip install .