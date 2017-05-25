from pytest import raises

from qqq.util import check_all_same_length


def test_check_all_same_length():

    assert check_all_same_length() == 0
    assert check_all_same_length([1, 2, 3], [4, 5, 6])
    with raises(ValueError):
        check_all_same_length([1, 2], [1, 2, 3])
    with raises(TypeError):
        check_all_same_length([1, 2, 3], None)
    with raises(TypeError):
        check_all_same_length([1, 2, 3], 55)

    assert check_all_same_length([1, 2, 3], None, allow_none=True) == 3
