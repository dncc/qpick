ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

.DEFAULT_GOAL := build/qpick

.PHONY: build/qpick
build/qpick:
	cargo build --release
	cargo build --release --manifest-path ./bin/Cargo.toml --verbose
	cd python_bindings && python setup.py install

.PHONY: build/dep
build/dep:
	sudo apt-get install libffi-dev
	cd /raid && curl https://sh.rustup.rs -sSf | sh -s -- -y
	$(shell source $(HOME)/.cargo/env)

.PHONY: build/pyqpick
build/pyqpick:
	cd python_bindings && python setup.py install

.PHONY: build/goqpick
build/goqpick:
	cd golang_service && make


