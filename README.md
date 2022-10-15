# Team Ekumen submission

## Submission data

> (i) how many num_episodes you want your solution to use in each level (mandatory)

We prepared our submission to be run using a single episode per level (`num_episodes = 1`).

> (ii) what method(s) you used and results you obtained (optional)

Our submission is an adapted implementation of the Model Predictive Contouring Control as described [here](https://rpg.ifi.uzh.ch/docs/TRO22_MPCC_Romero.pdf).

Due to performance limitations the algorithm is adapted not to plan the path in real-time, but instead to precalculate the path only once before the first episode. Due to uncertainty in the gate positions, the planned path is patched in real-time once the actual gate positions are known.

Also, for performance, the Casadi MPCC controller needs to be compiled. This will take a few seconds during the controller bring-up; a message will be printed on the terminal while this is happening.

From start to end, the execution of our controller should take less than a minute. You can find an example run in this video: https://www.youtube.com/watch?v=uAgVpQPQqDs

## Dependencies

Our submission uses the `networkx` python module.

We've added this dependency in the `pyproject.toml` file, so executing

```
pip install -e .
```

in the `safe-control-gym` folder of our fork should be enough to install this dependency.

The compilation stage of MPCC will also require a working `gcc` or `clang` installation.

## Usage instructions

To run our submission, the following commands should be enough.

```
cd competition/
python3 getting_started.py --overrides level0.yaml
python3 getting_started.py --overrides level1.yaml
python3 getting_started.py --overrides level2.yaml
python3 getting_started.py --overrides level3.yaml
```
