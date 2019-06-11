PROJECT_LOCAL_PATH := $(strip $(shell dirname "$(realpath $(lastword $(MAKEFILE_LIST)))"))
PROJECT_DIR := $(shell basename `pwd`)
HOME_REMOTE_PATH := ~/${USER}

PROJECT_REMOTE_PATH := /root/${USER}/${PROJECT_DIR}
#INDEX_REMOTE_PATH := /raid/${USER}/qpick-1b-test
INDEX_REMOTE_PATH := /raid/qpick/output

QPICK_BRANCH := master
TMUXW := 0

.PHONY: install/req
install/req:
	ssh root@${IP} "tmux new -d -s qpick || true && tmux select-window -t qpick:${TMUXW} \
										|| tmux new-window -n ${TMUXW} && \
					tmux send -t qpick:${TMUXW} 'apt-get update && apt-get install -y libxml2-dev python3-pip' ENTER && \
					tmux send -t qpick:${TMUXW} 'cd ${PROJECT_REMOTE_PATH} && source env.sh' ENTER && \
					tmux send -t qpick:${TMUXW} 'cd ${PROJECT_REMOTE_PATH}/scripts' ENTER && \
					tmux send -t qpick:${TMUXW} 'pip3 install pip --upgrade --ignore-installed' ENTER && \
					tmux send -t qpick:${TMUXW} 'PIP_INDEX_URL=\$$PIP_INDEX_URL \
									   PIP_TRUSTED_HOST=\$$PIP_TRUSTED_HOST \
									   pip3 install -v -q -r requirements.txt --ignore-installed' ENTER"

.PHONY: download/data
download/data:
	ssh root@${IP} "mkdir -p ${INDEX_REMOTE_PATH} && \
					tmux new -d -s qpick || true && tmux select-window -t qpick:${TMUXW} \
										|| tmux new-window -n ${TMUXW} && \
					tmux send -t qpick:${TMUXW} 'cd ${PROJECT_REMOTE_PATH} && source env.sh' ENTER && \
					tmux send -t qpick:${TMUXW} 'cd ${INDEX_REMOTE_PATH}' ENTER && \
					tmux send -t qpick:${TMUXW} 'aws s3 cp \$$INDEX_S3_DATA/qpick_input.woid.gz .' ENTER && \
					tmux send -t qpick:${TMUXW} 'aws s3 cp \$$INDEX_S3_DATA/gt/tq32.merged .' ENTER"

.PHONY: download/ws
download/ws:
	ssh root@${IP} "mkdir -p ${INDEX_REMOTE_PATH}/index && \
					tmux new -d -s qpick || true && tmux select-window -t qpick:${TMUXW} \
										|| tmux new-window -n ${TMUXW} && \
					tmux send -t qpick:${TMUXW} 'cd ${PROJECT_REMOTE_PATH} && source env.sh' ENTER && \
					tmux send -t qpick:${TMUXW} 'cd ${INDEX_REMOTE_PATH}/index' ENTER && \
					tmux send -t qpick:${TMUXW} 'aws s3 cp \$$INDEX_S3_QPICK/stopwords.txt .' ENTER && \
					tmux send -t qpick:${TMUXW} 'aws s3 cp \$$INDEX_S3_QPICK/config.json .' ENTER && \
					tmux send -t qpick:${TMUXW} 'aws s3 cp \$$INDEX_S3_QPICK/terms_relevance.fst .' ENTER"

.PHONY: qpick/rsync
qpick/rsync:
	ssh root@${IP} "mkdir -p ${PROJECT_REMOTE_PATH}"
	rsync -arvz \
		--exclude='.git' \
		--exclude='index' \
		--exclude='target' \
		--exclude='parts' \
		--exclude='.data' \
		--exclude='build' \
		--exclude='dist' \
		--exclude='*.egg-info' \
		--filter=': -.gitignore' \
	${PROJECT_LOCAL_PATH} root@${IP}:${HOME_REMOTE_PATH}

.PHONY: update/bin
update/bin:
	ssh root@${IP} "tmux send -t qpick:${TMUXW} 'cd ${PROJECT_REMOTE_PATH} && \
		   cargo build --release --manifest-path ./bin/Cargo.toml --verbose' ENTER && \
		   tmux send -t qpick:${TMUXW} 'cd ${INDEX_REMOTE_PATH} && \
		   cp ${PROJECT_REMOTE_PATH}/bin/target/release/qpick ${INDEX_REMOTE_PATH}' ENTER"

.PHONY: update/dev
update/dev: qpick/rsync update/bin

.PHONY: ssh/qpick/clone
ssh/qpick/clone:
	ssh root@${IP} "mkdir -p ${HOME_REMOTE_PATH} && \
		cd ${HOME_REMOTE_PATH} && \
		rm -rf qpick && git clone https://github.com/dncc/qpick.git && \
		cd qpick && git checkout ${QPICK_BRANCH}"

.PHONY: ssh/build/dep
ssh/build/dep:
	ssh root@${IP} "mkdir -p ${HOME_REMOTE_PATH} && \
		cd ${HOME_REMOTE_PATH}/qpick && \
		apt-get update && \
		apt-get install -y libffi-dev && \
		curl https://sh.rustup.rs -sSf | sh -s -- -y"

.PHONY: ssh/build/qpick
ssh/build/qpick:
	ssh root@${IP} "PATH=~/.cargo/bin:${PATH} && \
	cd ${HOME_REMOTE_PATH}/qpick && \
	cargo build --release && \
	cargo build --release --manifest-path ./bin/Cargo.toml --verbose"

.PHONY: ssh/build/pyqpick
ssh/build/pyqpick:
	ssh root@${IP} "PATH=~/.cargo/bin:${PATH} && \
	cd ${HOME_REMOTE_PATH}/qpick/python_bindings && \
	python3 setup.py install"

PHONY: ssh/install/test
ssh/install/test: qpick/rsync install/req download/data download/ws ssh/build/dep ssh/build/qpick ssh/build/pyqpick

PHONY: ssh/install/branch
ssh/install/qpick: ssh/qpick/clone ssh/build/dep ssh/build/qpick ssh/build/pyqpick

# ================ localhost =================
.PHONY: install/rust
install/rust:
	sudo apt-get install libffi-dev
	curl https://sh.rustup.rs -sSf | sh -s -- -y
	export PATH=~/.cargo/bin:${PATH}

.PHONY: clone/qpick
clone/qpick:
	ssh root@${IP} "mkdir -p ${HOME_REMOTE_PATH} && \
		cd ${HOME_REMOTE_PATH} && \
		rm -rf qpick && git clone https://github.com/dncc/qpick.git && \
		cd qpick && git checkout ${QPICK_BRANCH}"

.PHONY: build/qpick
build/qpick:
	PATH=~/.cargo/bin:${PATH}
	cargo build --release
	cargo build --release --manifest-path ./bin/Cargo.toml --verbose

.PHONY: build/pyqpick
build/pyqpick:
	cd python_bindings && python setup.py install

.PHONY: build/goqpick
build/goqpick:
	cd golang_service && make

.PHONY: install
install: build/qpick build/pyqpick


