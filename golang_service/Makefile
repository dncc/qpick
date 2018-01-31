ROOT_DIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

build:
	cd lib/goqpick && cargo build --release
	cp lib/goqpick/target/release/libgoqpick.so lib/
	go build -ldflags="-r $(ROOT_DIR)lib" main.go

run: build
	./main
