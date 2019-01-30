ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

build:
	cargo build --release
	cargo build --release --manifest-path ./bin/Cargo.toml --verbose
	cd golang_service && make
	cd python_bindings && python setup.py install

run: build
	./main
