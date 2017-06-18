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


class GFDSolver(GradientFreeSolver[A, B, DP_Arrs, P_Arrs, S]):
    '''
    A primitive finite differences solver using a gaussian perturbation.
    '''

    def fit(self, *, sigma=0.05, decay=0.01, **kwargs):
        super().fit(**kwargs)
        self.sigma_ = sigma
        self.decay_ = decay

    def get_perturbations(self, policy: P_Arrs) -> DP_Arrs:
        # XXX memallo optimize
        return [npr.normal(0, self.sigma_, size=p.shape) for p in policy]  # type: ignore  # noqa

    def apply_perturbation(self, policy: P_Arrs, delta: DP_Arrs) -> P_Arrs:
        return [p_arr + dp_arr for p_arr, dp_arr in zip(policy, delta)]  # type: ignore  # noqa

    def update(self, r0, policy, rollout_deltas, rewards):

        drs = np.array(rewards) - r0
        grad = []  # List[np.ndarray]

        for d_component in zip(*rollout_deltas):
            d_arr = np.stack([np.ravel(x) for x in d_component])

            g = self.lr_ * nlg.inv(d_arr.T @ d_arr) @ d_arr.T @ drs
            # print(g)
            grad.append(g)

        return [
            p + g.reshape(p.shape) - self.decay_ * p
            for p, g in zip(policy, grad)
        ]


if __name__ == '__main__':
    from qqq.rl.envs import HotCash
    from qqq.rl.actors import LinearActor

    env = HotCash(money_rate=0.5)
    actor = LinearActor()
    policy = actor.init_policy(env.observe().size, env.sample_action.size)
    solver = GFDSolver(env, sigma=0.1, lr=0.01)

    for i in range(100):
        policy = solver.train(policy, actor, epochs=100, n_base_rollouts=10, n_rollouts=100)
        print(policy[0].reshape((3, 3, 3, 4))[:, :, 0, 0])
        solver.rollout(policy, actor, draw=True)

