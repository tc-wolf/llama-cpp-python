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
	python3 -m build --sdist

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
	docker cp $$CONTAINER_ID:/llama_cpp/libllama.so - | pigz -9 > libllama.so.tgz ; \
	docker rm $$CONTAINER_ID

	# More cleanup
	yes | docker image prune
	
test:
	python3 -m pytest

docker:
	docker build -t llama-cpp-python:latest -f docker/simple/Dockerfile .

run-server:
	uvicorn --factory llama.server:app --host ${HOST} --port ${PORT}

clean:
	- cd vendor/llama.cpp && make clean
	- cd vendor/llama.cpp && rm libllama.so
	- rm -rf _skbuild
	- rm llama_cpp/*.so
	- rm llama_cpp/*.dylib
	- rm llama_cpp/*.metal
	- rm llama_cpp/*.dll
	- rm llama_cpp/*.lib

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
	docker \
	clean
