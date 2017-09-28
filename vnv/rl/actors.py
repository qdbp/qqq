from typing import List, Tuple

import numpy as np
import numpy.random as npr

from qqq.np import softmax
from qqq.rl.core import Actor


class LinearActor(
        Actor[
            np.ndarray,
            List[np.ndarray],
            np.ndarray
        ]):
    '''
    Actor implementing a basic linear function from observations to actions.
    '''

    def __init__(self, sample=True):
        self.sample_ = sample

    def init_policy(self, size, action_size) -> List[np.ndarray]:
        W = npr.random((size, action_size))
        b = npr.random(action_size)
        return [W, b]

    # TODO softmax
    def act(self, obs: np.ndarray, policy: List[np.ndarray]) -> np.ndarray:
        out = softmax(obs.ravel() @ policy[0] + policy[1])
        if self.sample_:
            return npr.multinomial(1, out)
        else:
            return out
