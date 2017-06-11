import os

import pytest

from qqq.qio_web import HTMLCutter as HC
from lxml.etree import XPathEvalError


def test_htmlcutter():
    with open(os.path.dirname(__file__) + '/test_qio_web_1.html', 'r') as f:
        s = f.read()

    # TEST 1
    hc = HC()\
        .xp('//p[@class="story-body-text story-content"]')\
        .strip('a')\
        .text()

    state = hc.cut(s)

    s5 = 'Mr. Mumford appeared in Federal District Court here on Friday wearing a dark suit and a matching yellow tie and pocket handkerchief. He said little more than “Yes, sir” in answer to questions from Judge John T. Fowlkes.'
    assert state[5] == s5

    # TEST 2
    hc = HC().xp('//time[@class="dateline"]/@datetime')
    state = hc.cut(s)

    assert state[0] == '2014-08-14T16:10:30-04:00'

    hc = HC()\
        .xp('//p')\
        .text()\
        .re(r'(c).(a)')\
        .sel(1)

    state = hc.cut(s)
    assert all([s == 'a' for s in state])

    with pytest.raises(XPathEvalError):
        opc1 = HC().xp('//p[@fail="fail]')
    with pytest.raises(XPathEvalError):
        opc2 = HC().xp('//p[@fail="fail"')


if __name__ == '__main__':
    # test_htmlcutter()
    pytest.main([__file__])
