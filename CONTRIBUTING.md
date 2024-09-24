# Getting Started

To get started, check out the open [Issues](https://github.com/kscalelabs/sim/issues).
We publish there the most pressing issues to contribute. Feel free to post a new one if you see 
an issue or you want to add an enhancement.

> [!NOTE]
> You should develop the backend using Python 3.10 or later.

When creating a new pull request please add the issue number.


## Lint and formatting
When you submit the PR we automatically run some checks using black, ruff and mypy.
You can check the logic [here](https://github.com/kscalelabs/sim/blob/master/pyproject.toml).
You can locally run commands below to test the formatting:
```
make format
make static-checks
```

## Adding a new robot
Creating a new embodiment is quite straightforward. The best way is to copy an existing robot and modify it.
The best starting point would be [stompymini](https://github.com/kscalelabs/sim/tree/master/sim/resources/stompymini).
1. Create a folder with a new robot [here](https://github.com/kscalelabs/sim/tree/master/sim/resources).
2. Add `joint.py` file setting up basic properties and joint configuration - see an [example](https://github.com/kscalelabs/sim/blob/master/sim/resources/stompymini/joints.py).
3. Update [height](https://github.com/kscalelabs/sim/blob/aa97ddfddcaadcac64d4f83d986548cc7fc451a6/sim/resources/stompymini/joints.py#L102) and default rotation of your humanoid.
4. Add a new embodiment configuration and environment [here](https://github.com/kscalelabs/sim/tree/master/sim/envs/humanoids).
5. Add a new embodiment to the [registry](https://github.com/kscalelabs/sim/blob/master/sim/envs/__init__.py).
To run things make sure that the robot name in your config file is the same as the folder name.
```
- sim
    - resources
        - NEW_HUMANOID
            - joints.py
            - urdf
            - ...
    - envs
        - humanoids
            - NEW_HUMANOID_config.py
            - NEW_HUMANOID_env.py
```
and kick off the training:
```
python sim/train.py --env-id NEW_HUMANOID --num-envs 4
```

## Making it stand
1. The best way to start is to make your new robot stand. In this case you want to comment out
the rewards that are responsible for making the robot walk. See an example of a [reward config](https://github.com/kscalelabs/sim/blob/aa97ddfddcaadcac64d4f83d986548cc7fc451a6/sim/envs/humanoids/stompymini_config.py#L163).
2. If the robot flies away, inspect your joint limits. During training, we introduce a lot of noise to the default joint positions
as well as masses. You might need either adapt the limits or the [noise level](https://github.com/kscalelabs/sim/blob/aa97ddfddcaadcac64d4f83d986548cc7fc451a6/sim/envs/humanoids/stompy_config.py#L213).
3. Isaac Sim often hits velocities nans when some joints are hitting their limits - you can change the [engine parameters](https://github.com/kscalelabs/sim/blob/aa97ddfddcaadcac64d4f83d986548cc7fc451a6/sim/envs/humanoids/stompy_config.py#L108) to fix this issue.
4. The revolute joints cannot have 0 velocity in the URDF definition - otherwise, engine will go nans as well.
5. Observe the reward for orientation and default joint position. The model should just farm these two rewards.
6. If the robot still struggles at keeping the standing pose, you can also change the urdf definition using the [script](https://github.com/kscalelabs/sim/blob/master/sim/scripts/create_fixed_torso.py) to fix the upper body or change joint limits.

## Making it walk
We set up the logic so that your new robot should start to walk with basic configuration after some modifications.
1. To get things going it's best to start from the good standing pose, with knees slightly bent.
See an example [Stompy's pose](https://github.com/kscalelabs/sim/blob/aa97ddfddcaadcac64d4f83d986548cc7fc451a6/sim/resources/stompymini/joints.py#L113).
2. The [gait reward](https://github.com/kscalelabs/sim/blob/aa97ddfddcaadcac64d4f83d986548cc7fc451a6/sim/envs/humanoids/stompy_config.py#L146) is the most crucial single reward. You have to adapt the hyperparameters to your robot design to get it to work.
3. If the robots tends to jump, use only one limb you will have to adapt the overall rewards schema to enforce proper behavior.
4. If the setup is correct, you should see positive results after first 200-300 epochs. However, the robust policy will requires 2000-3000 epochs.




