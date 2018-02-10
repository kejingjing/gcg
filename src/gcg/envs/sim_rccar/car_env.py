import sys
from math import pi
from collections import OrderedDict
import cv2
import numpy as np
import os

from direct.showbase.DirectObject import DirectObject
from direct.showbase.ShowBase import ShowBase
from panda3d.bullet import BulletBoxShape
from panda3d.bullet import BulletHelper
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletVehicle
from panda3d.bullet import BulletWorld
from panda3d.bullet import ZUp
from panda3d.core import AmbientLight
from panda3d.core import BitMask32
from panda3d.core import Point3
from panda3d.core import TransformState
from panda3d.core import Vec3
from panda3d.core import Vec4
from panda3d.core import loadPrcFileData

from gcg.envs.env_spec import EnvSpec
from gcg.envs.sim_rccar.panda3d_camera_sensor import Panda3dCameraSensor
from gcg.envs.spaces.box import Box
from gcg.envs.spaces.discrete import Discrete
from gcg.data.logger import logger


class CarEnv(DirectObject):
    def __init__(self, params={}):
        self._params = params
        if 'random_seed' in self._params:
            np.random.seed(self._params['random_seed'])
        self._use_vel = self._params.get('use_vel', True)
        self._run_as_task = self._params.get('run_as_task', False)
        self._do_back_up = self._params.get('do_back_up', False)
        self._use_depth = self._params.get('use_depth', False)
        self._use_back_cam = self._params.get('use_back_cam', False)

        self._collision_reward_only = self._params.get('collision_reward_only', False)
        self._collision_reward = self._params.get('collision_reward', -10.0)
        self._obs_shape = self._params.get('obs_shape', (64, 36))
        self._steer_limits = params.get('steer_limits', (-30., 30.))
        self._speed_limits = params.get('speed_limits', (-4.0, 4.0))
        self._motor_limits = params.get('motor_limits', (-0.5, 0.5))
        self._fixed_speed = (self._speed_limits[0] == self._speed_limits[1] and self._use_vel)
        if not self._params.get('visualize', False):
            loadPrcFileData('', 'window-type offscreen')

        # Defines base, render, loader

        try:
            ShowBase()
        except:
            pass
        
        base.setBackgroundColor(0.0, 0.0, 0.0, 1)

        # World
        self._worldNP = render.attachNewNode('World')
        self._world = BulletWorld()
        self._world.setGravity(Vec3(0, 0, -9.81))
        self._dt = params.get('dt', 0.25)
        self._step = 0.05

        # Vehicle
        shape = BulletBoxShape(Vec3(0.6, 1.0, 0.25))
        ts = TransformState.makePos(Point3(0., 0., 0.25))
        self._vehicle_node = BulletRigidBodyNode('Vehicle')
        self._vehicle_node.addShape(shape, ts)
        self._mass = self._params.get('mass', 10.) 
        self._vehicle_node.setMass(self._mass)
        self._vehicle_node.setDeactivationEnabled(False)
        self._vehicle_node.setCcdSweptSphereRadius(1.0)
        self._vehicle_node.setCcdMotionThreshold(1e-7)
        self._vehicle_pointer = self._worldNP.attachNewNode(self._vehicle_node)

        self._world.attachRigidBody(self._vehicle_node)

        self._vehicle = BulletVehicle(self._world, self._vehicle_node)
        self._vehicle.setCoordinateSystem(ZUp)
        self._world.attachVehicle(self._vehicle)
        self._addWheel(Point3( 0.3,  0.5, 0.07), True,  0.07)
        self._addWheel(Point3(-0.3,  0.5, 0.07), True,  0.07)
        self._addWheel(Point3( 0.3, -0.5, 0.07), False, 0.07)
        self._addWheel(Point3(-0.3, -0.5, 0.07), False, 0.07)

        # Camera
        size = self._params.get('size', [160, 90])
        hfov = self._params.get('hfov', 120)
        near_far = self._params.get('near_far', [0.1, 100.])
        self._camera_sensor = Panda3dCameraSensor(
            base,
            color=not self._use_depth,
            depth=self._use_depth,
            size=size,
            hfov=hfov,
            near_far=near_far,
            title='front cam')
        self._camera_node = self._camera_sensor.cam
        self._camera_node.setPos(0.0, 0.5, 0.375)
        self._camera_node.lookAt(0.0, 6.0, 0.0)
        self._camera_node.reparentTo(self._vehicle_pointer)

        if self._use_back_cam:
            self._back_camera_sensor = Panda3dCameraSensor(
                base,
                color=not self._use_depth,
                depth=self._use_depth,
                size=size,
                hfov=hfov,
                near_far=near_far,
                title='back cam')

            self._back_camera_node = self._back_camera_sensor.cam
            self._back_camera_node.setPos(0.0, -0.5, 0.375)
            self._back_camera_node.lookAt(0.0, -6.0, 0.0)
            self._back_camera_node.reparentTo(self._vehicle_pointer)
        
        # Car Simulator
        self._des_vel = None
        self._setup()
        
        # Input
        self.accept('escape', self._doExit)
        self.accept('r', self.reset)
        self.accept('f1', self._toggleWireframe)
        self.accept('f2', self._toggleTexture)
        self.accept('f3', self._view_image)
        self.accept('f5', self._doScreenshot)
        self.accept('q', self._forward_0)
        self.accept('w', self._forward_1)
        self.accept('e', self._forward_2)
        self.accept('a', self._left)
        self.accept('s', self._stop)
        self.accept('x', self._backward)
        self.accept('d', self._right)
        self.accept('m', self._mark)

        self._steering = 0.0       # degree
        self._engineForce = 0.0
        self._brakeForce = 0.0
        self._env_time_step = 0
        self._p = self._params.get('p', 1.25) 
        self._d = self._params.get('d', 0.0)
        self._last_err = 0.0
        self._curr_time = 0.0
        self._accelClamp = self._params.get('accelClamp', 0.5)
        self._engineClamp = self._accelClamp * self._mass
        self._collision = False

        self._setup_spec()

        self.spec = EnvSpec(
            observation_im_space=self.observation_im_space,
            action_space=self.action_space,
            action_selection_space=self.action_selection_space,
            observation_vec_spec=self.observation_vec_spec,
            action_spec=self.action_spec,
            action_selection_spec=self.action_selection_spec,
            goal_spec=self.goal_spec)

        if self._run_as_task:
            self._mark_d = 0.0
            taskMgr.add(self._update_task, 'updateWorld')
            base.run()

    def _setup_spec(self):
        self.action_spec = OrderedDict()
        self.action_selection_spec = OrderedDict()
        self.observation_vec_spec = OrderedDict()
        self.goal_spec = OrderedDict()

        self.action_spec['steer'] = Box(low=-45., high=45.)
        self.action_selection_spec['steer'] = Box(low=self._steer_limits[0], high=self._steer_limits[1])

        if self._use_vel:
            self.action_spec['speed'] = Box(low=-4., high=4.)
            self.action_space = Box(low=np.array([self.action_spec['steer'].low[0], self.action_spec['speed'].low[0]]),
                                    high=np.array([self.action_spec['steer'].high[0], self.action_spec['speed'].high[0]]))

            self.action_selection_spec['speed'] = Box(low=self._speed_limits[0], high=self._speed_limits[1])
            self.action_selection_space = Box(low=np.array([self.action_selection_spec['steer'].low[0],
                                                            self.action_selection_spec['speed'].low[0]]),
                                              high=np.array([self.action_selection_spec['steer'].high[0],
                                                             self.action_selection_spec['speed'].high[0]]))
        else:
            self.action_spec['motor'] = Box(low=-self._accelClamp, high=self._accelClamp)
            self.action_space = Box(low=np.array([self.action_spec['steer'].low[0], self.action_spec['motor'].low[0]]),
                                    high=np.array([self.action_spec['steer'].high[0], self.action_spec['motor'].high[0]]))

            self.action_selection_spec['motor'] = Box(low=self._motor_limits[0], high=self._motor_limits[1])
            self.action_selection_space = Box(low=np.array([self.action_selection_spec['steer'].low[0],
                                                            self.action_selection_spec['motor'].low[0]]),
                                              high=np.array([self.action_selection_spec['steer'].high[0],
                                                             self.action_selection_spec['motor'].high[0]]))

        assert (np.logical_and(self.action_selection_space.low >= self.action_space.low - 1e-4,
                               self.action_selection_space.high <= self.action_space.high + 1e-4).all())

        self.observation_im_space = Box(low=0, high=255, shape=tuple(self._get_observation()[0].shape))
        self.observation_vec_spec['coll'] = Discrete(1)
        self.observation_vec_spec['heading'] = Box(low=0, high=2 * 3.14)
        self.observation_vec_spec['speed'] = Box(low=-4.0, high=4.0)

    @property
    def _base_dir(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

    @property
    def horizon(self):
        return np.inf
   
    @property
    def max_reward(self):
        return np.inf

    # _____HANDLER_____

    def _doExit(self):
        sys.exit(1)

    def _toggleWireframe(self):
        base.toggleWireframe()

    def _toggleTexture(self):
        base.toggleTexture()

    def _doScreenshot(self):
        base.screenshot('Bullet')

    def _forward_0(self):
        self._des_vel = 1
        self._brakeForce = 0.0

    def _forward_1(self):
        self._des_vel = 2
        self._brakeForce = 0.0

    def _forward_2(self):
        self._des_vel = 4
        self._brakeForce = 0.0

    def _stop(self):
        self._des_vel = 0.0
        self._brakeForce = 0.0

    def _backward(self):
        self._des_vel = -4
        self._brakeForce = 0.0

    def _right(self):
        self._steering = np.min([np.max([-30, self._steering - 5]), 0.0])

    def _left(self):
        self._steering = np.max([np.min([30, self._steering + 5]), 0.0])

    def _view_image(self):
        from matplotlib import pyplot as plt
        image = self._camera_sensor.observe()[0]
        if self._use_depth:
            plt.imshow(image[:, :, 0], cmap='gray')
        else:
            def rgb2gray(rgb):
                return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

            image = rgb2gray(image)
            im = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA)  # TODO how does this deal with aspect ratio
            plt.imshow(im.astype(np.uint8), cmap='Greys_r')
        plt.show()

    def _mark(self):
        self._mark_d = 0.0

    # Setup
    
    def _setup(self):
        self._setup_map()
        self._place_vehicle()
        self._setup_light()
        self._setup_restart_pos()

    def _setup_map(self):
        if hasattr(self, '_model_path'):
            # Collidable objects
            self._setup_collision_object(self._model_path)
        else:
            ground = self._worldNP.attachNewNode(BulletRigidBodyNode('Ground'))
            shape = BulletPlaneShape(Vec3(0, 0, 1), 0)
            ground.node().addShape(shape)
            ground.setCollideMask(BitMask32.allOn())
            self._world.attachRigidBody(ground.node())

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

    def _setup_restart_pos(self):
        self._restart_index = 0
        self._restart_pos = self._default_restart_pos()

    def _next_restart_pos_hpr(self):
        num = len(self._restart_pos)
        if num == 0:
            return None, None
        else:
            pos_hpr = self._restart_pos[self._restart_index]
            self._restart_index = (self._restart_index + 1) % num
            return pos_hpr[:3], pos_hpr[3:]

    def _setup_light(self):
#        alight = AmbientLight('ambientLight')
#        alight.setColor(Vec4(0.5, 0.5, 0.5, 1))
#        alightNP = render.attachNewNode(alight)
#        render.clearLight()
#        render.setLight(alightNP)
        pass
    
    # Vehicle
    def _default_pos(self):
        return (0.0, 0.0, 0.3)

    def _default_hpr(self):
        return (0.0, 0.0, 0.0)

    def _default_restart_pos(self):
        return [self._default_pos() + self._default_hpr()]

    def _get_speed(self):
        vel = self._vehicle.getCurrentSpeedKmHour() / 3.6
        return vel

    def _get_heading(self):
        h = np.array(self._vehicle_pointer.getHpr())[0]
        ori = h * (pi / 180.)
        while (ori > 2 * pi):
            ori -= 2 * pi
        while(ori < 0):
            ori += 2 * pi
        return ori

    def _update(self, dt=1.0, coll_check=True):
        self._vehicle.setSteeringValue(self._steering, 0)
        self._vehicle.setSteeringValue(self._steering, 1)
        self._vehicle.setBrake(self._brakeForce, 0)
        self._vehicle.setBrake(self._brakeForce, 1)
        self._vehicle.setBrake(self._brakeForce, 2)
        self._vehicle.setBrake(self._brakeForce, 3)
        if dt >= self._step:
            # TODO maybe change number of timesteps
#            for i in range(int(dt/self._step)):
            if self._des_vel is not None:
                vel = self._get_speed()
                err = self._des_vel - vel
                d_err = (err - self._last_err) / self._step
                self._last_err = err
                self._engineForce = np.clip(self._p * err + self._d * d_err, -self._accelClamp, self._accelClamp) * self._mass
            self._vehicle.applyEngineForce(self._engineForce, 0)
            self._vehicle.applyEngineForce(self._engineForce, 1)
            self._vehicle.applyEngineForce(self._engineForce, 2)
            self._vehicle.applyEngineForce(self._engineForce, 3)
            for _ in range(int(dt/self._step)):    
                self._world.doPhysics(self._step, 1, self._step)
            self._collision = self._is_contact()
        elif self._run_as_task:
            self._curr_time += dt
            if self._curr_time > 0.05:
                if self._des_vel is not None:
                    vel = self._get_speed()
                    self._mark_d += vel * self._curr_time
                    print(vel, self._mark_d, self._is_contact())
                    err = self._des_vel - vel
                    d_err = (err - self._last_err) / 0.05
                    self._last_err = err
                    self._engineForce = np.clip(self._p * err + self._d * d_err, -self._accelClamp, self._accelClamp) * self._mass
                self._curr_time = 0.0
                self._vehicle.applyEngineForce(self._engineForce, 0)
                self._vehicle.applyEngineForce(self._engineForce, 1)
                self._vehicle.applyEngineForce(self._engineForce, 2)
                self._vehicle.applyEngineForce(self._engineForce, 3)
            self._world.doPhysics(dt, 1, dt)
            self._collision = self._is_contact()
        else:
            raise ValueError("dt {0} s is too small for velocity control".format(dt))

    def _stop_car(self):
        self._steering = 0.0
        self._engineForce = 0.0
        self._vehicle.setSteeringValue(0.0, 0)
        self._vehicle.setSteeringValue(0.0, 1)
        self._vehicle.applyEngineForce(0.0, 0)
        self._vehicle.applyEngineForce(0.0, 1)
        self._vehicle.applyEngineForce(0.0, 2)
        self._vehicle.applyEngineForce(0.0, 3)
        
        if self._des_vel is not None:
            self._des_vel = 0
        
        self._vehicle_node.setLinearVelocity(Vec3(0.0, 0.0, 0.0))
        self._vehicle_node.setAngularVelocity(Vec3(0.0, 0.0, 0.0))
        for i in range(self._vehicle.getNumWheels()):
            wheel = self._vehicle.getWheel(i)
            wheel.setRotation(0.0)
        self._vehicle_node.clearForces()

    def _place_vehicle(self, pos=None, hpr=None):
        if pos is None:
            pos = self._default_pos()
        if hpr is None:
            hpr = self._default_hpr()
        self._vehicle_pointer.setPos(pos[0], pos[1], pos[2])
        self._vehicle_pointer.setHpr(hpr[0], hpr[1], hpr[2])
        self._stop_car()

    def _addWheel(self, pos, front, radius=0.25):
        wheel = self._vehicle.createWheel()
        wheel.setChassisConnectionPointCs(pos)
        wheel.setFrontWheel(front)
        wheel.setWheelDirectionCs(Vec3(0, 0, -1))
        wheel.setWheelAxleCs(Vec3(1, 0, 0))
        wheel.setWheelRadius(radius)
        wheel.setMaxSuspensionTravelCm(40.0)
        wheel.setSuspensionStiffness(40.0)
        wheel.setWheelsDampingRelaxation(2.3)
        wheel.setWheelsDampingCompression(4.4)
        wheel.setFrictionSlip(1e2)
        wheel.setRollInfluence(0.1)

    # Task

    def _update_task(self, task):
        dt = globalClock.getDt()
        self._update(dt=dt)
        self._get_observation()
        return task.cont

    # Helper functions

    def _get_observation(self):
        self._obs = self._camera_sensor.observe()
        observation = []
        observation.append(self.process(self._obs[0], self._obs_shape))
        if self._use_back_cam:
            self._back_obs = self._back_camera_sensor.observe()
            observation.append(self.process(self._back_obs[0], self._obs_shape))
        observation_im = np.concatenate(observation, axis=2)
        coll = self._collision
        heading = self._get_heading()
        speed = self._get_speed()
        observation_vec = np.array([coll, heading, speed])
        return observation_im, observation_vec

    def _get_goal(self):
        return np.array([])

    def process(self, image, obs_shape):
        if self._use_depth:
            im = np.reshape(image, (image.shape[0], image.shape[1]))
            if im.shape != obs_shape:
                im = cv2.resize(im, obs_shape, interpolation=cv2.INTER_AREA)
            return im.astype(np.uint8)
        else:
            image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
            im = cv2.resize(image, obs_shape, interpolation=cv2.INTER_AREA) #TODO how does this deal with aspect ratio
            # TODO might not be necessary
            im = np.expand_dims(im, 2)
            return im.astype(np.uint8)

    def _get_reward(self):
        if self._collision_reward_only:
            if self._collision:
                reward = self._collision_reward
            else:
                reward = 0.0
        else:
            if self._collision:
                reward = self._collision_reward
            else:
                reward = self._get_speed()
        assert (reward <= self.max_reward)
        return reward

    def _get_done(self):
        return self._collision
    
    def _get_info(self):
        info = {}
        info['pos'] = np.array(self._vehicle_pointer.getPos())
        info['hpr'] = np.array(self._vehicle_pointer.getHpr())
        info['vel'] = self._get_speed()
        info['coll'] = self._collision
        info['env_time_step'] = self._env_time_step
        return info
    
    def _back_up(self):
        assert(self._use_vel)
#        logger.debug('Backing up!')
        self._params['back_up'] = self._params.get('back_up', {}) 
        back_up_vel = self._params['back_up'].get('vel', -1.0) 
        self._des_vel = back_up_vel
        back_up_steer = self._params['back_up'].get('steer', (-5.0, 5.0))
        self._steering = np.random.uniform(*back_up_steer)
        self._brakeForce = 0.
        duration = self._params['back_up'].get('duration', 3.0)
        self._update(dt=duration)
        self._des_vel = 0.
        self._steering = 0.
        self._update(dt=duration)
        self._brakeForce = 0.

    def _is_contact(self):
        result = self._world.contactTest(self._vehicle_node)
        return result.getNumContacts() > 0

    # Environment functions

    def reset(self, pos=None, hpr=None, hard_reset=False):
        if self._do_back_up and not hard_reset and \
                pos is None and hpr is None:
            if self._collision:
                self._back_up()
        else:
            if hard_reset:
                logger.debug('Hard resetting!')
            if pos is None and hpr is None:
                pos, hpr = self._next_restart_pos_hpr()
            self._place_vehicle(pos=pos, hpr=hpr)
        self._collision = False
        self._env_time_step = 0
        return self._get_observation(), self._get_goal()

    def step(self, action):
        self._steering = action[0]
        if action[1] == 0.0:
            self._brakeForce = 1000.
        else:
            self._brakeForce = 0.
        if self._use_vel:
            # Convert from m/s to km/h
            self._des_vel = action[1]
        else:
            self._engineForce = self._mass * action[1]

        self._update(dt=self._dt)
        observation = self._get_observation()
        goal = self._get_goal()
        reward = self._get_reward() 
        done = self._get_done()
        info = self._get_info()
        self._env_time_step += 1
        return observation, goal, reward, done, info

if __name__ == '__main__':
    params = {'visualize': True, 'run_as_task': True}
    env = CarEnv(params)
