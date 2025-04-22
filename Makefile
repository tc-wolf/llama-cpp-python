update:
	poetry install
	git submodule update --init --recursive

update.vendor:
	cd vendor/llama.cpp && git pull origin master

deps:
	python3 -m pip install --upgrade pip
	python3 -m pip install -e ".[all]"

build:
	python3 -m pip install --verbose -e .

build.debug:
	python3 -m pip install \
		--verbose \
		--config-settings=cmake.verbose=true \
		--config-settings=logging.level=INFO \
		--config-settings=install.strip=false  \
		--config-settings=cmake.args="-DCMAKE_BUILD_TYPE=Debug;-DCMAKE_C_FLAGS='-ggdb -O0';-DCMAKE_CXX_FLAGS='-ggdb -O0'" \
		--editable .

build.debug.extra:
	python3 -m pip install \
		--verbose \
		--config-settings=cmake.verbose=true \
		--config-settings=logging.level=INFO \
		--config-settings=install.strip=false  \
		--config-settings=cmake.args="-DCMAKE_BUILD_TYPE=Debug;-DCMAKE_C_FLAGS='-fsanitize=address -ggdb -O0';-DCMAKE_CXX_FLAGS='-fsanitize=address -ggdb -O0'" \
		--editable .

build.cuda:
	CMAKE_ARGS="-DGGML_CUDA=on" python3 -m pip install --verbose -e .

build.openblas:
	CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" python3 -m pip install --verbose -e .

build.blis:
	CMAKE_ARGS="-DGGML_BLAS=on -DGGML_BLAS_VENDOR=FLAME" python3 -m pip install --verbose -e .

build.metal:
	CMAKE_ARGS="-DGGML_METAL=on" python3 -m pip install --verbose -e .

build.vulkan:
	CMAKE_ARGS="-DGGML_VULKAN=on" python3 -m pip install --verbose -e .

build.kompute:
	CMAKE_ARGS="-DGGML_KOMPUTE=on" python3 -m pip install --verbose -e .

build.sycl:
	CMAKE_ARGS="-DGGML_SYCL=on" python3 -m pip install --verbose -e .

build.rpc:
	CMAKE_ARGS="-DGGML_RPC=on" python3 -m pip install --verbose -e .

build.sdist:
	python3 -m build --sdist --verbose

deploy.pypi:
	python3 -m twine upload dist/*

deploy.gh-docs:
	mkdocs build
	mkdocs gh-deploy

COMMIT := $(shell git rev-parse --short HEAD)

deploy.docker:
	# Make image with commit in name
	docker build -t openblas_server_$(COMMIT) .

	# Run image and immediately exit (just want to create the container)
	docker run openblas_server_$(COMMIT) bash

	# Get container ID, copy server tarball + libllama.so tarball, and delete
	# temp container
	CONTAINER_ID=$$(docker ps -lq --filter ancestor=openblas_server_$(COMMIT)) ; \
	echo Container ID: $$CONTAINER_ID ; \
	docker cp $$CONTAINER_ID:/root/dist/llama-cpp-py-server - | pigz -9 > llama-cpp-py-server.tgz ; \
	docker rm $$CONTAINER_ID

	# More cleanup
	yes | docker image prune

# Build standalone server, may want to do in fresh venv to avoid bloat
deploy.pyinstaller.mac:
	# CPU must be aarch64 and OS is MacOS
	@if [ `uname -m` != "arm64" ]; then echo "Must be on aarch64"; exit 1; fi
	@if [ `uname` != "Darwin" ]; then echo "Must be on MacOS"; exit 1; fi
	@echo "Building and installing with proper env vars for aarch64-specific ops"

	# This still builds with metal support (I think b/c GGML_NATIVE=ON). Not an
	# issue since can still run Q4_0 models w/ repacking support on CPU if `-ngl 0`.
	CMAKE_BUILD_TYPE="Release" \
	CMAKE_ARGS="-DGGML_METAL=OFF -DGGML_LLAMAFILE=OFF -DGGML_BLAS=OFF \
	-DGGML_NATIVE=ON -DGGML_CPU_AARCH64=ON" \
	python3 -m pip install -v -e .[server,pyinstaller]
	@server_path=$$(python -c 'import llama_cpp.server; print(llama_cpp.server.__file__)' | sed s/init/main/) ; \
	echo "Server path: $$server_path" ; \
	base_path=$$(python -c 'from llama_cpp._ggml import libggml_base_path; print(str(libggml_base_path))') ; \
	echo "Base path: $$base_path" ; \
	pyinstaller -DF $$server_path \
	--add-data $$base_path:llama_cpp/lib \
	-n llama-cpp-py-server

test:
	python3 -m pytest --full-trace -v

docker:
	docker build -t llama-cpp-python:latest -f docker/simple/Dockerfile .

run-server:
	python3 -m llama_cpp.server --model ${MODEL}

clean:
	- cd vendor/llama.cpp && make clean
	- cd vendor/llama.cpp && rm libllama.so
	- rm -rf _skbuild
	- rm llama_cpp/lib/*.so
	- rm llama_cpp/lib/*.dylib
	- rm llama_cpp/lib/*.metal
	- rm llama_cpp/lib/*.dll
	- rm llama_cpp/lib/*.lib

.PHONY: \
	update \
	update.vendor \
	build \
	build.cuda \
	build.opencl \
	build.openblas \
	build.sdist \
	deploy.pypi \
	deploy.gh-docs \
	deploy.docker \
	deploy.pyinstaller.mac \
	docker \
	clean
