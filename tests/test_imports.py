# super botch tests

def test_hit_import():
    try:
        from hierarchy_transformers import HierarchyTransformer
    except:
        assert 1 == 0 # fails
        return
    # else:
    assert 1 == 1
    return

def test_ont_import():
    try:
        from OnT.OnT import OntologyTransformer
    except:
        assert 1 == 0 # fails
        return
    # else:
    assert 1 == 1
    return