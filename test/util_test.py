from pytest import raises


def test_check_all_same_length():
    from vnv.util import check_all_same_length

    assert check_all_same_length() == 0
    assert check_all_same_length([1, 2, 3], [4, 5, 6])
    with raises(ValueError):
        check_all_same_length([1, 2], [1, 2, 3])
    with raises(TypeError):
        check_all_same_length([1, 2, 3], None)
    with raises(TypeError):
        check_all_same_length([1, 2, 3], 55)

    assert check_all_same_length([1, 2, 3], None, allow_none=True) == 3


def test_kws():
    from vnv.util import sift_kwargs, kwsift

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
