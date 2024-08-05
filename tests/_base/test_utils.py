from furax._base.utils import DefaultIdentityDict


def test_default_identity_dict() -> None:
    d = DefaultIdentityDict({'a': 'b'})
    assert d['a'] == 'b'
    assert 'c' not in d
    assert d['c'] == 'c'
