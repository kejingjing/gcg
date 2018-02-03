import os
from panda3d.bullet import BulletHelper
from panda3d.bullet import BulletRigidBodyNode
from panda3d.core import BitMask32

from gcg.envs.sim_rccar.square_env import SquareEnv

class SquareClutteredEnv(SquareEnv):
    def __init__(self, params={}):
        params.setdefault('model_path', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/square_cluttered.egg'))

        SquareEnv.__init__(self, params=params)

    def _setup_collision_object(self, path, pos=(0.0, 0.0, 0.0), hpr=(0.0, 0.0, 0.0), scale=1):
        visNP = loader.loadModel(path)
        visNP.clearModelNodes()
        visNP.reparentTo(render)
        visNP.setPos(pos[0], pos[1], pos[2])
        visNP.setHpr(hpr[0], hpr[1], hpr[2])
        visNP.set_scale(scale, scale, scale)
        bodyNPs = BulletHelper.fromCollisionSolids(visNP, True)
        for bodyNP in bodyNPs:
            bodyNP.reparentTo(render)
            if not ('wall' in bodyNP.name or 'ground' in bodyNP.name):
                bodyNP.setPos(pos[0], pos[1], pos[2] + 0.3)
            else:
                bodyNP.setPos(pos[0], pos[1], pos[2])
            bodyNP.setHpr(hpr[0], hpr[1], hpr[2])
            bodyNP.set_scale(scale, scale, scale)
            if isinstance(bodyNP.node(), BulletRigidBodyNode):
                bodyNP.node().setMass(0.0)
                bodyNP.node().setKinematic(True)
                bodyNP.setCollideMask(BitMask32.allOn())
                self._world.attachRigidBody(bodyNP.node())
            else:
                print("Issue")

if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': True, 'do_back_up': True, 'hfov': 120}
    env = SquareClutteredEnv(params)
