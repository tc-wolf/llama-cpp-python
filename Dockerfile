FROM ubuntu:22.04

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0

# Install the package
RUN apt update && apt install -y ninja-build build-essential pkg-config python3 python3-pip git
RUN python3 -m pip install --upgrade pip cmake scikit-build setuptools pyinstaller

COPY . .

# Install to /usr so that easily findable by cmake
RUN mv /OpenBLAS /opt/OpenBLAS && cd /opt/OpenBLAS && make install PREFIX=/usr/ && cd /

# Have to disable GGML_LLAMAFILE for Q4_0_4_4 quantization
RUN PKG_CONFIG_PATH="/opt/OpenBLAS/install/lib/pkgconfig" CMAKE_ARGS="-DGGML_BLAS=ON;-DGGML_BLAS_VENDOR=OpenBLAS;-DGGML_LLAMAFILE=OFF;-DCMAKE_C_FLAGS=-march=armv8.2-a+crypto+fp16+rcpc+dotprod -mcpu=cortex-a78c+crypto+noprofile+nossbs+noflagm+nopauth -mtune=cortex-a78c" pip install -e .[server]

RUN cd /root && pyinstaller -DF /llama_cpp/server/__main__.py \
    --add-data /usr/lib/libopenblas.so:. \
    --add-data /llama_cpp/lib/libllama.so:llama_cpp/lib \
    --add-data /llama_cpp/lib/libggml.so:llama_cpp/lib \
    -n llama-cpp-py-server
