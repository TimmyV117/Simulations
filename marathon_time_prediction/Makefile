# Makefile for creating Conda environment and installing packages

# Conda environment name
ENV_NAME = simulation_env2017

# Path to requirements.txt file
REQUIREMENTS_FILE = requirements.txt

.PHONY: create-env install-packages

create-env:
    @conda create --name $(ENV_NAME) python=3.6

install-packages: create-env
    @conda activate $(ENV_NAME) && \
    pip install -r $(REQUIREMENTS_FILE)

