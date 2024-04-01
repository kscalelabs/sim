# K-Scale Sim Library

## Getting Started

1. Clone this repository:

```bash
git clone git@github.com:kscalelabs/sim.git
cd sim
```

2. Install the Isaac Docker image:

```bash
./sim/scripts/install_isaac_dependencies.sh
```

3. Start the Isaac Docker image in Headless mode (assuming you're working in a server environment):

```bash
./sim/scripts/install_isaac_dependencies.sh
# Inside the Docker container
./runheadless.native.sh -v
```

### Notes

1. After cloning Isaac Gym, sometimes the bindings mysteriously disappear. To fix this, update the submodule:

```bash
git submodule update --init --recursive
```
