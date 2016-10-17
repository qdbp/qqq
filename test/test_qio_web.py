import os

import pytest 

from qqq.qio_web import HTMLCutter
from lxml.etree import XPathEvalError


def test_htmlcutter():
    with open(os.path.dirname(__file__) + '/test_qio_web_1.html', 'r') as f:
        s = f.read()

    ### TEST 1
    opc = [('xpath', '//p[@class="story-body-text story-content"]'),
           ('strip', 'a'),
           ('text', None)]

    wc = HTMLCutter(opc)
    state = wc.cut(s)
    s5 = 'Mr. Mumford appeared in Federal District Court here on Friday wearing a dark suit and a matching yellow tie and pocket handkerchief. He said little more than “Yes, sir” in answer to questions from Judge John T. Fowlkes.'
    assert state[5] == s5

    ### TEST 2
    opc = [('xp', '//time[@class="dateline"]/@datetime')]
    wc = HTMLCutter(opc)
    state = wc.cut(s)

    assert state[0] == '2014-08-14T16:10:30-04:00'

    # TEST 3
    opc = [('_fail_op', None)]
    with pytest.raises(ValueError):
        wc = HTMLCutter(opc)

    # TEST 4
    opc = [('xpath', '//@class'),
           ('xpath', './/p')]
    with pytest.raises(TypeError):
        wc = HTMLCutter(opc)

    opc = [('xpath', '//p'),
           ('text', ''),
           ('re', r'(c).(a)'),
           ('sel', 1)]
    state = HTMLCutter(opc).cut(s)
    assert all([s == 'a' for s in state])

    opc1 = [('xpath', '//p[@fail="fail]')]
    opc2 = [('xpath', '//p[@fail="fail"')]
    with pytest.raises(XPathEvalError):
        HTMLCutter(opc1)
    with pytest.raises(XPathEvalError):
        HTMLCutter(opc2)


if __name__ == '__main__':
    # test_htmlcutter()
    pytest.main([__file__])
