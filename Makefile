# Makefile

define HELP_MESSAGE
kscale-sim-library

# Installing

1. Create a new Conda environment: `conda create --name kscale-sim-library python=3.8.19`
2. Activate the environment: `conda activate kscale-sim-library`
3. Install the package: `make install-dev`

# Running Tests

1. Run autoformatting: `make format`
2. Run static checks: `make static-checks`
3. Run unit tests: `make test`

endef
export HELP_MESSAGE

all:
	@echo "$$HELP_MESSAGE"
.PHONY: all

# ------------------------ #
#          Train           #
# ------------------------ #

train-one-vis:
	@python -m sim.humanoid_gym.train --task stompy_ppo --run_name v1 --num_envs 1

train-many-vis:
	@python -m sim.humanoid_gym.train --task stompy_ppo --run_name v1 --num_envs 16

train:
	@python -m sim.humanoid_gym.train --task stompy_ppo --run_name v1 --num_envs 4096 --headless

play:
	@python -m sim.humanoid_gym.play --task stompy_ppo --run_name v1

# ------------------------ #
#          Build           #
# ------------------------ #

install:
	@pip install --verbose -e .
.PHONY: install

install-dev:
	@pip install --verbose -e '.[dev]'
.PHONY: install

install-third-party:
	@git submodule update --init --recursive
	@cd third_party/isaacgym/python/ && pip install --verbose -e .
	@cd third_party/humanoid-gym && pip install --verbose -e .
	@cd third_party/IsaacGymEnvs && pip install --verbose -e .

install-third-party-external:
	@git submodule update --init third_party/humanoid-gym
	@git submodule update --init third_party/IsaacGymEnvs
	@cd ${ISAACGYM_PATH}/python/ && pip install --verbose -e .
	@cd third_party/humanoid-gym && pip install --verbose -e .
	@cd third_party/IsaacGymEnvs && pip install --verbose -e .

build-ext:
	@python setup.py build_ext --inplace
.PHONY: build-ext

clean:
	rm -rf build dist *.so **/*.so **/*.pyi **/*.pyc **/*.pyd **/*.pyo **/__pycache__ *.egg-info .eggs/ .ruff_cache/
.PHONY: clean

# ------------------------ #
#       Static Checks      #
# ------------------------ #

format:
	@isort --profile black sim
	@black sim
	@ruff format sim
.PHONY: format

format-cpp:
	@clang-format -i $(shell find . -name '*.cpp' -o -name '*.h')
	@cmake-format -i $(shell find . -name 'CMakeLists.txt' -o -name '*.cmake')
.PHONY: format-cpp

static-checks:
	@isort --profile black --check --diff sim
	@black --diff --check sim
	@ruff check sim
	@mypy --install-types --non-interactive sim
.PHONY: lint

# ------------------------ #
#        Unit tests        #
# ------------------------ #

test:
	python -m pytest
.PHONY: test
