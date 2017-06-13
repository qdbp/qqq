# TODO: optimize and cythonize
'''
Basic classes and functions to work with basic RL environments.
'''
from abc import ABC, abstractmethod
import pickle
from typing import Tuple

import numpy as np
import numpy.linalg as nlg
import numpy.random as npr

from qqq.qlog import get_logger

log = get_logger(__file__)


def fd_grad(dps, r0, rs):
    '''
    Finite  difference gradient estimate.
    '''

    gs = []
    for dp in dps:
        dr = np.asarray(rs) - r0
        dp = np.stack([np.ravel(x) for x in dp])

        gs.append(nlg.inv((dp.T @ dp)) @ dp.T @ dr)

    return gs


def fd_grad_gauss(sigma, dps, r0, rs):
    '''
    Finite difference gradient estimate under gaussian perturbation.

    Assume diagonal covariance matrix of fixed radius to save computation.
    '''

    gs = []
    for dp in dps:
        dr = np.asarray(rs) - r0
        dp = np.stack([np.ravel(x) for x in dp])

        gs.append((1/sigma) * np.eye(dp.shape[1]) @ dp.T @ dr)

    return gs


class MandatoryAttributeError(Exception):
    pass


class RLEnv(ABC):

    def __init__(self, **kwargs):
        self.fit(**kwargs)
        self.init_state()
        self.reset_state()

        if not hasattr(self, 'state'):
            raise MandatoryAttributeError(
                f'The `state` attribute should be set on {self} by one of the '
                'state initialization methods.')

    @property
    def size(self):
        return self.state.size

    @abstractmethod
    def draw(self) -> str:
        '''
        Return a string representation of the environment.

        Should be suitable for human visualization.
        '''

    @abstractmethod
    def step(self, a: np.ndarray) -> Tuple[float, bool]:
        '''
        Advance the environment with action `a`.
        '''

    @abstractmethod
    def fit(self) -> None:
        '''
        Assigns mutable parameters to environment.

        No parameters should be assigned outside of this function.
        Should be safe to call between rollouts. Should be idempotent.

        Guaranteed to be called at least once before any init_ methods.

        Attributes set by fit should follow the sklearn trailing underscore
        convention.
        '''

    @abstractmethod
    def init_state(self) -> None:
        '''
        Allocation memory for state objects and does one-off init.

        Run only once on instantiation.
        '''

    @abstractmethod
    def reset_state(self) -> None:
        '''
        Resets the state between rollouts or other experiment repeats.

        Should not reinvoke one-time-init code.
        '''


class RLSolver(ABC):

    def __init__(self, e: int, n: int, z: int, env: RLEnv, **kwargs) -> None:
        '''
        Args:
            e: number of epochs
            n: number of rollouts per epoch
            z: steps per rollout
            env: the RLEnv object on which learning will be done
        '''
        self.e = e
        self.n = n
        self.z = z

        self.env = env

        self.fit(**kwargs)
        self.init_policy()

    @abstractmethod
    def fit(self):
        '''
        Assigns mutable parameters to the solver.

        No parameters should be assigned outside of this function.

        Attributes set by fit should follow the sklearn trailing underscore
        convention.
        '''

    @abstractmethod
    def init_policy(self):
        '''
        Initialize the policy.

        Run once on instantiation. Must create a `policy` attribute.
        '''

    @abstractmethod
    def act(self, state, policy):
        '''
        Performs one step with the current policy in the current env state.
        '''

    @abstractmethod
    def train(self):
        '''
        Trains the solver with the `fit`ted parameters.
        '''

    def rollout(self, policy) -> float:
        reward = 0.
        for t in range(self.z):
            r_t, alive = self.env.step(self.act(self.env.state, policy))
            reward += r_t/self.z
            if not alive:
                break
        return reward


class FDSolver(RLSolver):

    def fit(self, h=10, lr=0.05, sigma=0.2):
        self.h_ = h  # hidden state dimension
        self.lr_ = lr
        self.sigma_ = sigma

    def _perturb(self):
        return [npr.normal(size=m.shape, scale=self.sigma_)
                for m in self.policy]

    def train(self):
        log.info(f'starting training')
        av_rs = []
        for e in range(self.e):
            r0 = self.rollout(self.policy)
            rs, dps = [], []
            for rix in range(self.n):
                dp = self._perturb()
                pp = [m + dm for m, dm in zip(self.policy, dp)]

                rs.append(self.rollout(pp))
                dps.append(dp)

            gs = fd_grad_gauss(self.sigma_, zip(*dps), r0, rs)
            self.policy = [x + self.lr_ * dx.reshape(x.shape)
                           for x, dx in zip(self.policy, gs)]

            av_rs.append(np.mean(rs))
            if not (e % 25):
                av_r = np.mean(av_rs)
                av_rs = []

                log.info(f'epoch {e}: attained reward {av_r:.3f}')

        with open('policy.p', 'wb') as f:
            pickle.dump((self.env, self.policy), f)
            log.info('saved environment and policy')
