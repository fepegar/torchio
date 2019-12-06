def to_tuple(value, n=1):
    """
    to_tuple(1, n=1) -> (1,)
    to_tuple(1, n=3) -> (1, 1, 1)

    If value is an iterable, n is ignored and tuple(value) is returned
    to_tuple((1,), n=1) -> (1,)
    to_tuple((1, 2), n=1) -> (1, 2)
    to_tuple([1, 2], n=3) -> (1, 2)
    """
    try:
        iter(value)
        value = tuple(value)
    except TypeError:
        value = n * (value,)
    return value
