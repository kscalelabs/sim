# Getting Started

To get started, check out the open [Issues](https://github.com/kscalelabs/sim/issues).
We publish there the most pressing issues to contribute. Feel free to post a new one if you see 
an issue or you want to add an enhancement.

> [!NOTE]
> You should develop the backend using Python 3.10 or later.

When creating a new pull request please add the issue number.

## Adding new robot
Adding new embodiment is very straightforward:
1. Create a folder with a new robot [here](https://github.com/kscalelabs/sim/tree/master/sim).
2. Add joint.py file setting up basic properties and join configuration - see an [example](https://github.com/kscalelabs/sim/blob/master/sim/resources/stompymini/joints.py).
3. Add the new embodiment configuration and environment [here](https://github.com/kscalelabs/sim/tree/master/sim/envs).
4. Add the new embodiment to the [registry](https://github.com/kscalelabs/sim/blob/master/sim/envs/__init__.py).

We set up the logic so that your new robot should start to walk with basic configuration.

## Lint and formatting
When you submit the PR we automatically run some checks using black, ruff and mypy.
You can check the logic [here](https://github.com/kscalelabs/sim/blob/master/pyproject.toml).
You can locally run commands below to test the formatting:
```
make format
make static-checks
```
