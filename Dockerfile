FROM ubuntu:22.04

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0

# Install the package
RUN apt update && apt install -y ninja-build build-essential pkg-config python3 python3-pip git
RUN python3 -m pip install --upgrade pip cmake scikit-build setuptools pyinstaller

COPY . .

# Install to /usr so that easily findable by cmake
RUN mv /OpenBLAS /opt/OpenBLAS && cd /opt/OpenBLAS && make install PREFIX=/usr/ && cd /

RUN PKG_CONFIG_PATH="/opt/OpenBLAS/install/lib/pkgconfig" CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_LLAMAFILE=1" pip install -e .[server]

RUN cd /root && pyinstaller -DF /llama_cpp/server/__main__.py \
    --add-data /opt/OpenBLAS/:. \
    --add-data /llama_cpp/:. \
    -n llama-cpp-py-server
