# TODO: optimize and cythonize
'''
Basic classes and functions to work with basic RL environments.
'''
from abc import ABC, abstractmethod, abstractproperty
import pickle
from time import sleep
from typing import NewType, TypeVar, Generic, List
from typing import Tuple

import numpy as np
import numpy.linalg as nlg
import numpy.random as npr

from qqq.qlog import get_logger
from qqq.util import sift_kwargs

log = get_logger(__file__)

A = TypeVar('A')  # actions
B = TypeVar('B')  # observations
P = TypeVar('P')  # policies
S = TypeVar('S')  # state
# XXX: very hacky
DP = TypeVar('DP')  # delta policy
GP = TypeVar('GP')  # grad of policy


class MandatoryAttributeError(Exception):
    pass


class Actor(Generic[A, B, P]):
    '''
    Class taking a policy and an observation into an action vector.
    '''

    @abstractmethod
    def act(self, observation: B, policy: P) -> A:
        '''
        Performs an action given observation and policy.
        '''


class DiffActor(Actor[A, B, P], Generic[A, B, P, GP]):
    '''
    An actor with a differentiable policy.
    '''

    @abstractmethod
    def grad(self, policy: P) -> GP:
        '''
        The gradient of the policy with respect to its parameters.
        '''


class RLEnv(ABC, Generic[A, B, S]):

    def __init__(self, **kwargs):
        self.fit(**kwargs)
        self.init_state()
        self.reset_state()

    @abstractproperty
    def state(self) -> S:
        '''
        Returns a view of the environment's state.

        Not a proper observation.
        '''

    @abstractproperty
    def sample_action(self) -> A:
        '''
        Returns a sample action object.
        '''

    @abstractmethod
    def draw(self) -> str:
        '''
        Return a string representation of the environment.

        Should be suitable for human visualization.
        '''

    @abstractmethod
    def step(self, a: A) -> Tuple[float, bool]:
        '''
        Advance the environment with action `a`.

        Returns the (reward, termination_flag, new_observation).
        '''

    def observe(self) -> B:
        '''
        Observe the environment to get an observation vector.
        '''
        return self._observe()

    @abstractmethod
    def _observe(self) -> B:
        pass

    @abstractmethod
    def _fit(self, **kwargs) -> None:
        '''
        Assigns mutable parameters to environment.

        No parameters should be assigned outside of this function.
        Should be safe to call between rollouts. Should be idempotent.

        Guaranteed to be called at least once before any init_ methods.

        Attributes set by fit should follow the sklearn trailing underscore
        convention.
        '''

    def fit(self, **kwargs) -> None:
        self._fit(**kwargs)

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


class RLSolver(ABC, Generic[A, B, P, S]):
    '''
    Base class for all solvers.

    Its methods cover the basics of hyperparameter and environment
    initialization, and adhere where possible to the sklearn approach
    '''

    def __init__(self, env: RLEnv[A, B, S], **kwargs) -> None:
        '''
        Args:
            e: number of epochs
            n: number of rollouts per epoch
            z: steps per rollout
            env: the RLEnv object on which learning will be done
        '''
        self.env = env

        self.fit(**kwargs)

    def reset_env(self) -> None:
        self.env.reset_state()

    def fit(self, **kwargs):
        '''
        Assigns mutable parameters to the solver.

        No parameters should be assigned outside of this function.

        Attributes set by fit should follow the sklearn trailing underscore
        convention.
        '''
        if kwargs:
            log.warning(
                f'unknown keyword arguments {set(kwargs.keys())} '
                f'in {self.__class__}'
            )


# XXX: mixinize
class RolloutMixin(RLSolver[A, B, P, S], Generic[A, B, P, S]):
    '''
    Proto-policy gradient solver base class.
    '''

    @sift_kwargs
    def rollout(
            self,
            policy: P,
            actor: Actor[A, B, P],
            *,
            max_length=100,
            draw=False,
            reset=True) -> float:
        '''
        Performs a rollout of a policy with a given actor.

        Returns the total reward.
        '''

        if reset:
            self.env.reset_state()

        reward = 0.
        for t in range(max_length):

            obs = self.env.observe()
            action = actor.act(obs, policy)
            r_t, alive = self.env.step(action)

            reward += r_t

            if draw:
                self.env.draw()
                sleep(0.1)

            if not alive:
                break
        return reward


class GradientFreeSolver(RolloutMixin[A, B, P, S], Generic[A, B, DP, P, S]):
    '''
    Base class for gradient-free solvers.

    Finite differences, CEM, CMA-ES, etc, should be implementable on top
    of this.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, lr=0.05, **kwargs):
        super().fit(**kwargs)
        self.lr_ = lr

    def train(
            self,
            policy: P,
            actor: Actor[A, B, P],
            epochs=100,
            n_base_rollouts=50,
            n_rollouts=250,
            rollout_length=100,
            **kw) -> P:
        '''
        Trains the policy.

        Takes an initial policy and an actor, returns the final policy.
        '''

        log.info(f'starting training')
        av_rs = []
        for e in range(epochs):

            r0 = 0.
            for i in range(n_base_rollouts):
                r0 += self.rollout(
                    policy, actor, max_length=rollout_length, **kw
                ) / n_base_rollouts

            rewards = []  # type: List[float]
            deltas = []  # type: List[DP]

            for rix in range(n_rollouts):
                dp = self.get_perturbations(policy)
                p_rollout = self.apply_perturbation(policy, dp)
                rewards.append(
                    self.rollout(
                        p_rollout, actor, max_length=rollout_length, **kw
                    )
                )
                deltas.append(dp)

            new_policy = self.update(r0, policy, deltas, rewards)

            av_rs.append(np.mean(rewards))
            if not (e % 25):
                av_r = np.mean(av_rs)
                av_rs = []

                log.info(f'epoch {e}: attained reward {av_r:.3f}')

            policy = new_policy

        with open('policy.p', 'wb') as f:
            pickle.dump((self.env, policy), f)
            log.info('saved environment and policy')

        return policy

    @abstractmethod
    def get_perturbations(self, policy: P) -> DP:
        '''
        Generate perturbations of the policy.
        '''

    @abstractmethod
    def apply_perturbation(self, policy: P, dp: DP) -> P:
        '''
        Generate a new policy from a perturbation.
        '''

    @abstractmethod
    def update(
            self,
            base_reward: float,
            policy: P,
            deltas: List[DP],
            rewards: List[float]) -> P:
        '''
        Return an updated policy from a collection of perturbations and
        rewards obtained from applying them to the current policy.

        Arguments:
            base_reward: reward of current policy
            policy: current policy
            deltas: perturbations applied to the policy
            rewards: rewards obtained by the perturbations
        '''
