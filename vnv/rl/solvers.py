from abc import abstractmethod
import pickle
from typing import Generic, TypeVar, List, NewType

import numpy as np
import numpy.linalg as nlg
import numpy.random as npr

from qqq.qlog import get_logger
from qqq.rl.core import GradientFreeSolver, A, B, P, S


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


P_Arrs = NewType('P_Arrs', List[np.ndarray])
DP_Arrs = NewType('DP_Arrs', List[np.ndarray])


class ESSolver(GradientFreeSolver[A, B, DP_Arrs, P_Arrs, S]):
    '''
    A basic single-threaded Evolution Strategies solver.

    Salimans, Tim, et al. "Evolution strategies as a scalable
    alternative to reinforcement learning." arXiv preprint arXiv:1703.03864
    (2017).

    Implements antithetic sampling and a rank transformation.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self._antithetic = False

    def fit(self, *, sigma=0.05, decay=0.01, **kwargs):
        super().fit(**kwargs)
        self.sigma_ = sigma
        self.decay_ = decay

    def _rank_transform(self, rewards: List[float]):
        return np.linspace(-len(rewards), len(rewards)) / len(rewards)

    def get_perturbations(self, policy: P_Arrs) -> DP_Arrs:
        # XXX memalloc optimize
        if not self._antithetic:
            self._perturbation = [
                npr.normal(0, self.sigma_, size=p.shape) for p in policy
            ]
            self._antithetic = True
            return self._perturbation
        else:
            self._antithetic = False
            return [-pert for pert in self._perturbation]

    def apply_perturbation(self, policy: P_Arrs, delta: DP_Arrs) -> P_Arrs:
        return [p_arr + dp_arr for p_arr, dp_arr in zip(policy, delta)]  # type: ignore  # noqa

    def update(self, policy, rollout_deltas, rewards) -> P_Arrs:

        drs = self._rank_transform(np.array(rewards))
        grad = []  # List[np.ndarray]

        for d_component in zip(*rollout_deltas):
            d_arr = np.stack([np.ravel(x) for x in d_component])

            g = self.lr_ * nlg.inv(d_arr.T @ d_arr) @ d_arr.T @ drs
            grad.append(g)

        return [  # type: ignore
            p + g.reshape(p.shape) - p * self.decay_
            for p, g in zip(policy, grad)
        ]

    def train(self, *args, **kwargs):
        return super().train(*args, n_base_rollouts=0, **kwargs)


class ESSolver(GradientFreeSolver[A, B, DP_Arrs, P_Arrs, S]):
    '''
    Evolution strategies solver.

    Following

    '''

    def __init__(self):
        pass

    def get_perturbations(self, policy: P_Arrs) -> DP_Arrs:
        pass

    def apply_perturbation(self, delta: DP_Arrs, policy: P_Arrs) -> P_Arrs:
        pass

    def update(


if __name__ == '__main__':
    from qqq.rl.envs import HotCash
    from qqq.rl.actors import LinearActor

    env = HotCash(money_rate=0.5)
    actor = LinearActor(sample=False)
    policy = actor.init_policy(env.observe().size, env.sample_action.size)
    solver = GFDSolver(env, sigma=1., lr=0.001)

    for i in range(10):
        policy = solver.train(
            policy,
            actor,
            epochs=200,
            n_base_rollouts=10,
            n_rollouts=50,
            rollout_length=10,
        )

        print(policy[0].reshape((3, 3, 3, 4))[:, :, 2, 0])
        solver.rollout(policy, actor, draw=True)
