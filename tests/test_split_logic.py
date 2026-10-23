def test_split_ids_disjoint():
    # Lightweight test of the intended invariant.
    train = {"A", "B"}
    val = {"C"}
    test = {"D"}
    assert not train.intersection(val)
    assert not train.intersection(test)
    assert not val.intersection(test)
