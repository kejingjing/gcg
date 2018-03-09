import os, copy
from panda3d.bullet import BulletHelper
from panda3d.bullet import BulletRigidBodyNode
from panda3d.core import BitMask32

from gcg.envs.sim_rccar.square_env import SquareEnv

class SquareClutteredEnv(SquareEnv):

    def __init__(self, params={}):
        self.use_alternative_restart_pos = params['use_alternative_restart_pos']  # bool
        SquareEnv.__init__(self, params=params)

    @property
    def _model_path(self):
        return os.path.join(self._base_dir, 'square_cluttered.egg')

    def _setup_collision_object(self, path, pos=(0.0, 0.0, 0.0), hpr=(0.0, 0.0, 0.0), scale=1, ignore_collision=False):
        visNP = loader.loadModel(path)
        visNP.clearModelNodes()
        visNP.reparentTo(render)
        visNP.setPos(pos[0], pos[1], pos[2])
        visNP.setHpr(hpr[0], hpr[1], hpr[2])
        visNP.set_scale(scale, scale, scale)
        if not ignore_collision:
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

    def _default_restart_pos(self):
        if self.use_alternative_restart_pos:

            l = [
                [20., -20., 0.3, 0.0, 0.0, 0.0],
                [20., -17., 0.3, 0.0, 0.0, 0.0],
                [20., -12., 0.3, 0.0, 0.0, 0.0],
                [20., -9., 0.3, 0.0, 0.0, 0.0],
                [20., -4., 0.3, 0.0, 0.0, 0.0],
                [20., 4., 0.3, 0.0, 0.0, 0.0],
                [20., 10., 0.3, 0.0, 0.0, 0.0],

                [20., 20., 0.3, 90.0, 0.0, 0.0],
                [14., 20., 0.3, 90.0, 0.0, 0.0],
                [6., 20., 0.3, 90.0, 0.0, 0.0],
                [1., 20., 0.3, 90.0, 0.0, 0.0],
                [-2., 20., 0.3, 90.0, 0.0, 0.0],
                [-8., 20., 0.3, 90.0, 0.0, 0.0],
                [-12., 20., 0.3, 90.0, 0.0, 0.0],

                [-20., 20., 0.3, 180.0, 0.0, 0.0],
                [-20., 15., 0.3, 180.0, 0.0, 0.0],
                [-20., 10., 0.3, 180.0, 0.0, 0.0],
                [-20., 5., 0.3, 180.0, 0.0, 0.0],
                [-20., 0., 0.3, 180.0, 0.0, 0.0],
                [-20., -5., 0.3, 180.0, 0.0, 0.0],
                [-20., -13., 0.3, 180.0, 0.0, 0.0],

                [-20., -20., 0.3, 270.0, 0.0, 0.0],
                [-15., -20., 0.3, 270.0, 0.0, 0.0],
                [-10., -20., 0.3, 270.0, 0.0, 0.0],
                [-5., -20., 0.3, 270.0, 0.0, 0.0],
                [-2., -20., 0.3, 270.0, 0.0, 0.0],
                [5., -20., 0.3, 270.0, 0.0, 0.0],
                [13., -20., 0.3, 270.0, 0.0, 0.0],
                ]

            for start in copy.copy(l):
                start[3] += 180.
                l.append(start)

            return l

        else:
            return SquareEnv._default_restart_pos(self)


if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': True, 'do_back_up': True, 'hfov': 120}
    env = SquareClutteredEnv(params)
