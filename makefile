BUILD_DIR ?= build
CUDA_ARCH ?= 80

CMAKE_FLAGS := \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_TESTING=ON \
	-DENABLE_CUDA=ON \
	-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)

.PHONY: all configure build test benchmark clean

all: build

configure:
	cmake -S . -B $(BUILD_DIR) $(CMAKE_FLAGS)

build: configure
	cmake --build $(BUILD_DIR) --parallel

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

benchmark: build
	./$(BUILD_DIR)/nbody_app

clean:
	rm -rf $(BUILD_DIR)