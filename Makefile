PROJECT_LOCAL_PATH := $(strip $(shell dirname "$(realpath $(lastword $(MAKEFILE_LIST)))"))
PROJECT_DIR := $(shell basename `pwd`)
HOME_REMOTE_PATH := ~/${USER}
PROJECT_REMOTE_PATH := ~/${USER}/${PROJECT_DIR}
QPICK_BRANCH := nget_long_query_exp

.DEFAULT_GOAL := build/fq

.PHONY: build/fq
build/fq:
	$(info Building ${PROJECT_DIR}...)

.PHONY: rsync
rsync:
	ssh root@${IP} "mkdir -p ${PROJECT_REMOTE_PATH}"
	rsync -arvz \
	${PROJECT_LOCAL_PATH}/ root@${IP}:${PROJECT_REMOTE_PATH}/

.PHONY: rsync/extraction
rsync/extraction:
	ssh root@${IP} "mkdir -p ${HOME_REMOTE_PATH}/content_extraction"
	rsync -arvz \
	${PROJECT_LOCAL_PATH}/../content_extraction/ root@${IP}:${HOME_REMOTE_PATH}/content_extraction/


.PHONY: install/req
install/req:
	ssh root@${IP} "tmux new -d -s fq || true && tmux select-window -t fq:0 || tmux new-window -n 0 && \
					tmux send -t fq:0 'cd ${PROJECT_REMOTE_PATH} && apt-get install libxml2-dev' ENTER && \
					tmux send -t fq:0 'pip install pip --upgrade --ignore-installed' ENTER && \
					tmux send -t fq:0 'PIP_INDEX_URL=http://pypi.cliqz.discover:8080/simple/ \
									   PIP_TRUSTED_HOST=pypi.cliqz.discover \
									   pip install -v -q -r requirements.txt --ignore-installed' ENTER && \
					tmux send -t fq:0 'VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3 \
									   source virtualenvwrapper.sh && workon -c p3' ENTER"

.PHONY: install/redis
install/redis:
	ssh root@${IP} "curl -L 'http://download.redis.io/releases/redis-5.0.3.tar.gz' > redis-5.0.3.tar.gz \
    				&& tar xvzf redis-5.0.3.tar.gz \
    				&& cd redis-5.0.3 \
    				&& make \
    				&& make install"

.PHONY: download/ws
download/ws:
	ssh root@${IP} "cd ${PROJECT_REMOTE_PATH} && \
					S3=s3://cliqz-mapreduce/fresh_index/i18n/v1/word_statistics_keyvi && \
					aws s3 cp ${S3}/2019-01-21T11-40-27/word_statistics.kv ."

.PHONY: download/pages
download/pages:
	ssh root@${IP} "mkdir -p ${PROJECT_REMOTE_PATH}/pages && cd ${PROJECT_REMOTE_PATH}/pages && \
					S3=s3://cliqz-data-pipeline/test/partial_merged_snippets/urls-dragan/de/20190404/ && \
					aws s3 cp --recursive ${S3} ."

.PHONY: qpick/rsync
qpick/rsync:
	ssh root@${IP} "mkdir -p ${HOME_REMOTE_PATH}/qpick"
	rsync -arvz \
		--exclude='.git' \
		--exclude='index' \
		--exclude='target' \
		--exclude='parts' \
		--exclude='.data' \
		--exclude='scripts' \
		--exclude='build' \
		--exclude='dist' \
		--exclude='*.egg-info' \
		--filter=': -.gitignore' \
	${QPICK_LOCAL_PATH} root@${IP}:${HOME_REMOTE_PATH}/qpick/

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

