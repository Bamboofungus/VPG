# Vanilla Policy Gradient

:
## Introduction
A *relatively* simple implementation of Vanilla policy gradients from https://spinningup.openai.com/en/latest/algorithms/vpg.html
trained on the `lunarlander_v2` environment from gymnasium (https://gymnasium.farama.org/environments/box2d/lunar_lander)

In this environment, an agent tries to land a spacecraft inside the goal (denoted by the flags) but gets penalized for each 
timestep it fires it's thrusters or if it crashes into the moon's surface. To add a bit of randomness, the spacecraft is already moving
in a direction when the environment starts.

Uses some "tricks" over the original implementation of VPG:
- Entropy loss
- Generalized advantage estimation w/ advantage normalization
- Learning rate annealing
- Orthogonal initialization of weights

Empirically, advantage normalization seemed to boost performance the most by scaling down gradients followed
by entropy loss over epsilon-greedy for exploration. Additionally, environments are carried over between optimization steps instead
of being terminated like spinningup's implementation.

The model in `vpg.pth` was trained on 2,000,000 steps, but the average rewards remained mostly the same past ~400,000 steps, which
is probably due to known issues with VPG (getting trapped in local optima) that PPO (Proximal policy optimization) aimed to fix.

## Setup
Install the dependencies:
```python
pip install poetry
poetry run shell
poetry install
```
Run a training loop:
```python
python3 lunarlander_train.py
```
Hyperparameters are managed by Hydra in `/config/vpg.cfg` but can be overriden in the command line:
```python
python3 lunarlander_train.py epochs=20
```
Evaluate a saved model:
```python
python3 lunarlander_eval.py vpg.pth

## Additional references
https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
https://joschu.net/docs/thesis.pdf
https://andyljones.com/posts/rl-debugging.html
``````
