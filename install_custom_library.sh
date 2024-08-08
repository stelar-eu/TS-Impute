sudo apt-get update \
&& sudo apt-get install build-essential libopenmpi-dev libopenblas-dev liblapack-dev libmlpack-dev \
&& sudo apt-get install python3.8-dev \
&& sudo apt-get update \
&& sudo apt-get install ca-certificates gpg wget \
&& wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
&& UBUNTUCODENAME=$(lsb_release -cs) \
&& echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/' $UBUNTUCODENAME 'main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null \
&& sudo apt-get update \
&& sudo apt-get install cmake \
&& wget -N http://sourceforge.net/projects/arma/files/armadillo-10.8.2.tar.xz \
&& sudo rm -rf armadillo-10.8.2 \
&& tar -xvf armadillo-10.8.2.tar.xz \
&& cd armadillo-10.8.2 \
&& ./configure \
&& make \
&& sudo make install \
&& cd .. \
&& pip install pybind11==2.9.2 \
&& PYTHONENVLIB=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') \
&& PYTHONENVEXE=$(which python3) \
&& cd AlgoCollection \
&& sudo rm -rf build \
&& mkdir build \
&& cd build \
&& cmake .. -DPYTHON_LIBRARY_DIR=$PYTHONENVLIB -DPYTHON_EXECUTABLE=$PYTHONENVEXE -Dpybind11_DIR=$PYTHONENVLIB"/pybind11/share/cmake/pybind11" \
&& make \
&& make install \
&& cd .. \
&& cd .. \
&& pip install torch==2.1.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html \
&& pip install torch-cluster==1.6.3 torch-sparse==0.6.18 torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.2+cu121.html \
&& pip install .