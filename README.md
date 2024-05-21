# Vanilla Policy Gradient :robot:
## Introduction

A *relatively* simple implementation of Vanilla policy gradients from https://spinningup.openai.com/en/latest/algorithms/vpg.html
trained on Gymnasium's [Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander).

<div align="center">
  <img src="https://github.com/Bamboofungus/VPG/assets/40710895/96c3639c-cc39-4548-b083-08b30f823ac7" alt="VPG_Lunarlander_9500">
</div>


In Lunar Lander, an agent tries to land a spacecraft inside the goal (denoted by the flags) but gets penalized for each 
timestep it fires it's thrusters or if it crashes into the moon's surface. To add a bit of randomness, the spacecraft is already moving
in a direction when the environment starts.

Uses some "tricks" not in the original implementation of VPG that are used in PPO:
- Entropy loss to promote exploration
- Learning rate annealing
- Orthogonal initialization of weights

The model in `vpg.pth` was trained on 2,000,000 steps, but the average rewards remained the same past ~400,000 steps, which
is probably due to known issues with VPG (getting trapped in local optima).

## Running
Install the dependencies:
```sh
pip install poetry
poetry run shell
poetry install
```
Run a training loop:
```sh
python lunarlander_train.py
```
Hyperparameters are managed by Hydra in `/config/vpg.cfg` but can be overriden in the command line:
```sh
python lunarlander_train.py epochs=20
```
See a saved model interacting with the environment:
```sh
python lunarlander_eval.py vpg.pth
```

## Additional references
- https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details
- https://joschu.net/docs/thesis.pdf
- https://andyljones.com/posts/rl-debugging.html
