FROM ubuntu:20.04

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST=0.0.0.0

# Needs to be <= 2.31 to work with older Linux
RUN echo "Glibc version:\n" && /lib/aarch64-linux-gnu/libc.so.6

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    --no-install-recommends ninja-build pkg-config python3.9 \
    python3-pip git

# Install toolchain (GCC 11)
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y --no-install-recommends gcc-11 g++-11 make && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 \
    --slave /usr/bin/g++ g++ /usr/bin/g++-11 --slave /usr/bin/gcov gcov /usr/bin/gcov-11 \
    --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-11 \
    --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-11 \
    --slave /usr/bin/cpp cpp /usr/bin/cpp-11

RUN python3.9 -m pip install --upgrade pip cmake scikit-build-core[pyproject] setuptools pyinstaller

# Install the repo at current state
COPY . .

# Set HW specific flags for us
ENV march=armv8.2-a+crypto+fp16+rcpc+dotprod 
ENV mcpu=cortex-a78c+crypto+noprofile+nossbs+noflagm+nopauth
ENV mtune=cortex-a78c

ENV compiler_flags="-march=${march} -mcpu=${mcpu} -mtune=${mtune}"

# This is a release build that works (have to disable GGML_LLAMAFILE for Q4_0_4_4 quantization)
RUN CMAKE_ARGS="-DGGML_LLAMAFILE=OFF -DCMAKE_C_FLAGS='${compiler_flags}' -DCMAKE_CXX_FLAGS='${compiler_flags}' -DCMAKE_BUILD_TYPE=Release" python3.9 -m pip install -v -e .[server] 2>&1 | tee buildlog.txt

RUN cd /root && pyinstaller -DF /llama_cpp/server/__main__.py \
    --add-data /llama_cpp/lib/libllama.so:llama_cpp/lib \
    --add-data /llama_cpp/lib/libggml.so:llama_cpp/lib \
    -n llama-cpp-py-server
