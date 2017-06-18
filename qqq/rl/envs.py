import sys
from typing import Tuple

import numpy as np
import numpy.random as npr

from qqq.rl.core import RLEnv


class HotCash(RLEnv[np.ndarray, np.ndarray, np.ndarray]):
    '''
    Money spawns in an arena. Grab it! Don't touch the lava.

    A primitive gridworld meant to test implementation correctness of
    basic prototypes and experiments.
    '''

    lava_max = 127
    money_max = 5
    MONEY_PLANE = 0
    LAVA_PLANE = 1

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def sample_action(self):
        return np.zeros((4,))

    def _fit(self, size=10, money_rate=0.05, lava_rate=0.002, obs_size=3):
        self.size_ = size
        self.money_rate_ = money_rate
        self.lava_rate_ = lava_rate
        self.obs_size_ = obs_size

    def init_state(self):
        self._state = np.zeros(
            (self.size_ + 2, self.size_ + 2, 3), dtype=np.int8,
        )
        self._obs = np.zeros((self.obs_size_, self.obs_size_, 3))
        # fill in the walls
        self._state[:, :, -1] = 1
        self._state[1:-1, 1:-1, -1] = 0

        self._canvas = np.zeros((self.size_ + 2, self.size_ + 2), dtype='|S1')
        self._canvas[:] = '#'

    def reset_state(self):
        # clear all non-walls
        self._state[:, :, :-1] = 0
        self._x, self._y = self._random_xy()

        # initialize some money and lava
        for i in range(10):
            self._do_spawn()

    def _observe(self) -> np.ndarray:
        self._obs[:, :] =\
            self._state[self._x - 1:self._x + 2, self._y - 1:self._y + 2]

        out = self._obs.view()
        out.flags['WRITEABLE'] = False

        return out

    def _random_xy(self):
        return npr.randint(1, self.size_ + 1), npr.randint(1, self.size_ + 1)

    def _do_spawn(self):
        # decay money and lava
        self._state[:, :, :-1] = np.maximum(self._state[:, :, :-1] - 1, 0)

        # spawn new money
        for i in range(5):
            if npr.random() < self.money_rate_:
                x, y = self._random_xy()
                self._state[x, y, self.MONEY_PLANE] =\
                    npr.randint(self.money_max)

            # spawn new lava
            if npr.random() < self.lava_rate_:
                x, y = self._random_xy()
                self._state[x, y, self.LAVA_PLANE] = npr.randint(self.lava_max)

    def _move(self, ix: int) -> None:
        if ix == 0:
            self._x = np.maximum(self._x - 1, 1)
        elif ix == 1:
            self._x = np.minimum(self._x + 1, self.size_)
        elif ix == 2:
            self._y = np.maximum(self._y - 1, 1)
        else:
            self._y = np.minimum(self._y + 1, self.size_)

    def _money_check(self) -> Tuple[float, bool]:
        pos_vec = self._state[self._x, self._y]
        if pos_vec[self.LAVA_PLANE] > 0:
            return (0, False)
        else:
            # money value is not proportional to how long it has left
            return (float(pos_vec[self.MONEY_PLANE] > 0), True)

    def step(self, a: np.ndarray) -> Tuple[float, bool]:
        self._move(np.argmax(a))
        r, alive = self._money_check()
        self._do_spawn()
        return (r, alive)

    def draw(self):

        self._canvas[1:-1, 1:-1] = ' '
        self._canvas[self._state[:, :, self.MONEY_PLANE] > 0] = '$'
        self._canvas[self._state[:, :, self.LAVA_PLANE] > 0] = '&'
        self._canvas[self._x, self._y] = '@'

        sys.stdout.write(
            '\n'.join([''.join(
                [c.decode('ascii') for c in row]
            ) for row in self._canvas])
        )
        sys.stdout.write("\033[F" * (self.size_ + 1))
        sys.stdout.flush()


if __name__ == '__main__':
    env = HotCash()
    
    alive = True
    while True:
        env.draw()
        print(env.observe())
        r, alive = env.step(npr.random(size=(4,)))
