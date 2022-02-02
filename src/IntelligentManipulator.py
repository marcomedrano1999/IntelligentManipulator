"""
Intelligent manipulator code

Author: Marco Medrano
Date: 31/01/2022
"""


from email.errors import ObsoleteHeaderDefect
import robosuite
from robosuite.controllers import load_controller_config
import numpy as np

controller_config = load_controller_config(default_controller="OSC_POSE")

env = robosuite.make(
    "PickPlace",
    robots=["UR5e"],
    gripper_types="default",
    controller_configs=controller_config,
    has_renderer=True,
    render_camera="frontview",
    has_offscreen_renderer=False,
    control_freq=20,
    horizon=20000,
    use_object_obs=False,
    use_camera_obs=False,
    camera_names="agentview",
    camera_heights=84,
    camera_widths=84,
)

env.reset()

low, high = env.action_spec

for i in range(10000):
    action = np.random.uniform(low,high)
    obs, reward, done, _ = env.step(action)
    print(obs)
    print(reward)
    env.render()