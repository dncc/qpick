PROJECT_LOCAL_PATH := $(strip $(shell dirname "$(realpath $(lastword $(MAKEFILE_LIST)))"))
PROJECT_DIR := $(shell basename `pwd`)
HOME_REMOTE_PATH := ~/${USER}
PROJECT_REMOTE_PATH := ~/${USER}/${PROJECT_DIR}
INDEX_REMOTE_PATH := /raid/${USER}/qpick-1b-test
QPICK_BRANCH := i2q
TMUXW := 1

.PHONY: install/req
install/req:
	ssh root@${IP} "tmux new -d -s qpick || true && tmux select-window -t qpick:${TMUXW} \
										|| tmux new-window -n ${TMUXW} && \
					tmux send -t qpick:${TMUXW} 'cd ${PROJECT_REMOTE_PATH}/scripts' ENTER && \
					tmux send -t qpick:${TMUXW} 'apt-get install libxml2-dev' ENTER && \
					tmux send -t qpick:${TMUXW} 'pip install pip --upgrade --ignore-installed' ENTER && \
					tmux send -t qpick:${TMUXW} 'PIP_INDEX_URL=http://pypi.cliqz.discover:8080/simple/ \
									   PIP_TRUSTED_HOST=pypi.cliqz.discover \
									   pip install -v -q -r requirements.txt --ignore-installed' ENTER && \
					tmux send -t qpick:${TMUXW} 'export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3' ENTER && \
					tmux send -t qpick:${TMUXW} 'python3 pip install virtualenvwrapper' ENTER && \
					tmux send -t qpick:${TMUXW} 'source virtualenvwrapper.sh && workon -c p3' ENTER"

.PHONY: download/test/data
download/test/data:
	ssh root@${IP} "mkdir -p ${INDEX_REMOTE_PATH} && \
					tmux new -d -s qpick || true && tmux select-window -t qpick:${TMUXW}
										|| tmux new-window -n ${TMUXW} && \
					tmux send -t qpick:${TMUXW} 'cd ${PROJECT_REMOTE_PATH} && source index_s3.sh' ENTER && \
					tmux send -t qpick:${TMUXW} 'cd ${INDEX_REMOTE_PATH}' ENTER && \
					tmux send -t qpick:${TMUXW} 'aws s3 cp \$$INDEX_S3/qpick_input.txt .' ENTER && \
					tmux send -t qpick:${TMUXW} 'aws s3 cp \$$INDEX_S3/i2q/i2q.kv .' ENTER && \
					tmux send -t qpick:${TMUXW} 'aws s3 cp \$$INDEX_S3/gt/tq32.merged .' ENTER"

.PHONY: download/pages
download/pages:
	ssh root@${IP} "mkdir -p ${PROJECT_REMOTE_PATH}/pages && cd ${PROJECT_REMOTE_PATH}/pages && \
					S3=s3://cliqz-data-pipeline/test/partial_merged_snippets/urls-dragan/de/20190404/ && \
					aws s3 cp --recursive ${S3} ."

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

.PHONY: qpick/clone
qpick/clone:
	ssh root@${IP} "cd ${HOME_REMOTE_PATH} && \
					rm -rf qpick && \
					git clone https://github.com/dncc/qpick.git && cd qpick && git checkout ${QPICK_BRANCH}"

.PHONY: qpick/dep
qpick/dep:
	ssh root@${IP} "cd ${HOME_REMOTE_PATH}/qpick && \
	apt-get install libffi-dev && \
	curl https://sh.rustup.rs -sSf | sh -s -- -y"

.PHONY: qpick/build
qpick/build:
	ssh root@${IP} "PATH=~/.cargo/bin:${PATH} && \
	cd ${HOME_REMOTE_PATH}/qpick && \
	cargo build --release && \
	cargo build --release --manifest-path ./bin/Cargo.toml --verbose"

.PHONY: pyqpick/build
pyqpick/build:
	ssh root@${IP} "PATH=~/.cargo/bin:${PATH} && \
	cd ${HOME_REMOTE_PATH}/qpick/python_bindings && \
	python setup.py install"

PHONY: qpick/install
qpick/install: qpick/clone qpick/dep qpick/build pyqpick/build

PHONY: install
install: rsync install/req download/ws qpick/install


# ================ localhost =================
.PHONY: build/dep
build/dep:
	sudo apt-get install libffi-dev
	curl https://sh.rustup.rs -sSf | sh -s -- -y
	export PATH=~/.cargo/bin:${PATH}

.PHONY: build/qpick
build/qpick:
	PATH=~/.cargo/bin:${PATH}
	cargo build --release
	cargo build --release --manifest-path ./bin/Cargo.toml --verbose

.PHONY: build/pyqpick
build/pyqpick:
	cd python_bindings && python setup.py install

.PHONY: install
install: build/dep build/qpick build/pyqpick


.PHONY: build/goqpick
build/goqpick:
	cd golang_service && make

