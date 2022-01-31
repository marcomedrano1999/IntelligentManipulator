"""
Robot environment

Author: Marco Medrano
Date: 30/01/2022
"""

from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint
from mujoco_py import MjSim, MjViewer


class robotEnv():


    def __init__(self):
        # Create world env 
        self.world = MujocoWorldBase()

        # Create robot and gripper
        self.create_robot()

        # Create objs
        self.create_objs()

        # Create scene
        self.create_scene()
        
        # Run simulation
        model = self.world.get_model(mode="mujoco_py")
        self.sim = MjSim(model)
        self.viewer = MjViewer(self.sim)
        self.viewer.vopt.geomgroup[0] = 0 #disable visualization of collision mesh

        """
        for i in range(100000):
            self.sim.data.ctrl[:] = 0
            self.sim.step()
            self.viewer.render()
        """


    def create_robot(self):
        # Create robot model (in this case, a panda robot)
        mujoco_robot = Panda()

        # Create a gripper and attach it to the roibot
        gripper = gripper_factory('PandaGripper')
        mujoco_robot.add_gripper(gripper)

        # Add robot to world
        mujoco_robot.set_base_xpos([0,0,0])
        self.world.merge(mujoco_robot)

    def create_objs(self):
        # Add a sphere
        sphere = BallObject(name='sphere', size=[0.04], rgba=[0,0.5,0.5,1]).get_obj()
        sphere.set('pos','1.0 0 1.0')
        self.world.worldbody.append(sphere)

    def create_scene(self):
        # Create a table 
        mujoco_arena = TableArena()
        mujoco_arena.set_origin([0.8,0,0])
        self.world.merge(mujoco_arena)

    
def main():
    # Create env
    env = robotEnv()

    # Run simulation
    for i in range(100000):
            env.sim.data.ctrl[:] = 0
            env.sim.step()
            env.viewer.render()


if __name__=="__main__":
    main()
