# mypy: disable-error-code="import-untyped"
#!/usr/bin/env python
"""Setup script for the project."""

import re
from typing import List

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()


with open("sim/requirements.txt", "r", encoding="utf-8") as f:
    requirements: List[str] = f.read().splitlines()


with open("sim/requirements-dev.txt", "r", encoding="utf-8") as f:
    requirements_dev: List[str] = f.read().splitlines()


with open("sim/__init__.py", "r", encoding="utf-8") as fh:
    version_re = re.search(r"^__version__ = \"([^\"]*)\"", fh.read(), re.MULTILINE)
assert version_re is not None, "Could not find version in sim/__init__.py"
version: str = version_re.group(1)


setup(
    name="kscale-sim-library",
    version=version,
    description="Training models in simulation",
    author="Benjamin Bolte, Pawel Budzianowski, Allen Wu",
    url="https://github.com/kscalelabs/sim",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8, <3.9",
    install_requires=requirements,
    tests_require=requirements_dev,
    extras_require={"dev": requirements_dev},
)
