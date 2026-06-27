import os

import ogbench
import gymnasium as gym
import numpy as np


env = gym.make("antsoccer-twoants-arena-v0")

obs, info = env.reset(seed=0)

print("obs shape:", obs.shape)
print("action space:", env.action_space)
print("action dim:", env.action_space.shape)
print("model nq:", env.unwrapped.model.nq)
print("model nv:", env.unwrapped.model.nv)
print("model nu:", env.unwrapped.model.nu)
print("qpos shape:", env.unwrapped.data.qpos.shape)
print("qvel shape:", env.unwrapped.data.qvel.shape)

assert env.action_space.shape[0] == 16
assert env.unwrapped.model.nu == 16


# --------------------------------------------------
# Check initial positions
# --------------------------------------------------
ant1_xy, ant2_xy, ball_xy = env.unwrapped.get_two_ant_ball_xy()
print("initial:")
print("ant1_xy:", ant1_xy)
print("ant2_xy:", ant2_xy)
print("ball_xy:", ball_xy)


# --------------------------------------------------
# Optional: set clean starting positions
# --------------------------------------------------
env.unwrapped.set_two_ant_ball_xy(
    ant1_xy=np.array([-4.0, 0.0]),
    ant2_xy=np.array([4.0, 0.0]),
    ball_xy=np.array([0.0, 0.0]),
)

ant1_xy, ant2_xy, ball_xy = env.unwrapped.get_two_ant_ball_xy()
print("after manual set:")
print("ant1_xy:", ant1_xy)
print("ant2_xy:", ant2_xy)
print("ball_xy:", ball_xy)


# --------------------------------------------------
# Test: move only ant1
# --------------------------------------------------
before = env.unwrapped.get_two_ant_ball_xy()
print("before ant1 move:", before)

for _ in range(20):
    a = np.zeros(env.action_space.shape)
    a[:8] = env.action_space.sample()[:8]   # only ant1 gets action
    obs, reward, terminated, truncated, info = env.step(a)

after = env.unwrapped.get_two_ant_ball_xy()
print("after ant1 move:", after)


# --------------------------------------------------
# Reset positions again before checking ant2
# --------------------------------------------------
env.unwrapped.set_two_ant_ball_xy(
    ant1_xy=np.array([-4.0, 0.0]),
    ant2_xy=np.array([4.0, 0.0]),
    ball_xy=np.array([0.0, 0.0]),
)


# --------------------------------------------------
# Test: move only ant2
# --------------------------------------------------
before = env.unwrapped.get_two_ant_ball_xy()
print("before ant2 move:", before)

for _ in range(20):
    a = np.zeros(env.action_space.shape)
    a[8:] = env.action_space.sample()[8:]   # only ant2 gets action
    obs, reward, terminated, truncated, info = env.step(a)

after = env.unwrapped.get_two_ant_ball_xy()
print("after ant2 move:", after)


env.close()
print("SUCCESS")