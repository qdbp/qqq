import os

import pytest 

from qqq.qio_web import HTMLCutter as HC
from lxml.etree import XPathEvalError


def test_htmlcutter():
    with open(os.path.dirname(__file__) + '/test_qio_web_1.html', 'r') as f:
        s = f.read()

    ### TEST 1
    opc = [(HC.XPATH, '//p[@class="story-body-text story-content"]'),
           (HC.STRIP, 'a'),
           (HC.TEXT, None)]

    wc = HC(opc)
    state = wc.cut(s)
    s5 = 'Mr. Mumford appeared in Federal District Court here on Friday wearing a dark suit and a matching yellow tie and pocket handkerchief. He said little more than “Yes, sir” in answer to questions from Judge John T. Fowlkes.'
    assert state[5] == s5

    ### TEST 2
    opc = [(HC.XPATH, '//time[@class="dateline"]/@datetime')]
    wc = HC(opc)
    state = wc.cut(s)

    assert state[0] == '2014-08-14T16:10:30-04:00'

    # TEST 3
    opc = [('_fail_op', None)]
    with pytest.raises(ValueError):
        wc = HC(opc)

    # TEST 4
    opc = [(HC.XPATH, '//@class'),
           (HC.XPATH, './/p')]
    with pytest.raises(TypeError):
        wc = HC(opc)

    opc = [(HC.XPATH, '//p'),
           (HC.TEXT, ''),
           (HC.REGEX, r'(c).(a)'),
           (HC.SELECT, 1)]
    state = HC(opc).cut(s)
    assert all([s == 'a' for s in state])

    opc1 = [(HC.XPATH, '//p[@fail="fail]')]
    opc2 = [(HC.XPATH, '//p[@fail="fail"')]
    with pytest.raises(XPathEvalError):
        HC(opc1)
    with pytest.raises(XPathEvalError):
        HC(opc2)


if __name__ == '__main__':
    # test_htmlcutter()
    pytest.main([__file__])
