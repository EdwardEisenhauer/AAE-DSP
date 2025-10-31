IMAGE_NAME=opencv

.DEFAULT_GOAL := docker

venv : .venv/bin/activate ## Create Python virtual environment
.DELETE_ON_ERROR :
.venv/bin/activate :
	@echo "Creating virtual environment..."
	@python3 -m venv venv
.PHONY : venv

image : Dockerfile
	docker build --tag ${IMAGE_NAME} .
.PHONY : image

docker : image
	docker run \
		--rm \
		--interactive \
		--volume `pwd`:/data \
		${IMAGE_NAME} \
		python3 /data/tasks.py /data/resources/car1.jpg
.PHONY : docker
