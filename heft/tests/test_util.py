from heft.util import reverse_dict

def test_reverse_dict():
    d = {'a': (1, 2), 'b': (2, 3), 'c': ()}
    assert reverse_dict(d) == {1: ('a',), 2: ('a', 'b'), 3: ('b',)}
