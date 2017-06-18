from pytest import raises


def test_check_all_same_length():
    from qqq.util import check_all_same_length

    assert check_all_same_length() == 0
    assert check_all_same_length([1, 2, 3], [4, 5, 6])
    with raises(ValueError):
        check_all_same_length([1, 2], [1, 2, 3])
    with raises(TypeError):
        check_all_same_length([1, 2, 3], None)
    with raises(TypeError):
        check_all_same_length([1, 2, 3], 55)

    assert check_all_same_length([1, 2, 3], None, allow_none=True) == 3


def test_uql():
    from qqq.util import uql, UQLError
    d = {
        'A': {
            'a': ['A.a.0', 'A.a.1'],
            'b': [],
        },
        'B': {
            'c': {},
            'd': 'FOOSTRING',
            'e': 99,
        }
    }

    assert uql(d, 'B.c') == {}
    assert uql(d, 'C.a.z.z.99', 'def') == 'def'
    assert uql(d, 'A.b.1', 55) == 55
    assert uql(d, 'B.d.3') == 'S'

    with raises(UQLError):
        uql(d, 'A.a.foo', 99)
    with raises(UQLError):
        uql(d, 'B.d.z', 99)


def test_kws():
    from qqq.util import sift_kwargs, kwsift

    @sift_kwargs
    def foo(*, b, a=1):
        return b

    assert foo(c=1, b=2) == 2
    assert foo(**kwsift(dict(c=1, b=2), foo)) == 2

    class Foo:

        @sift_kwargs
        def foo(self, z=99):
            return z

    assert Foo().foo(b=10) == 99
