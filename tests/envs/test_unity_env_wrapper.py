import pytest

try:
    from envs.unity_env_wrapper import UnityEnvWrapper
except ImportError:
    UnityEnvWrapper = None


def test_imports():
    """Test that UnityEnvWrapper can be imported."""
    assert UnityEnvWrapper is not None


@pytest.mark.skipif(UnityEnvWrapper is None, reason="UnityEnvWrapper not available")
def test_wrapper_class_exists():
    """Test that UnityEnvWrapper class exists and has expected attributes."""
    # Test that the class has the expected methods
    assert hasattr(UnityEnvWrapper, '__init__')
    assert hasattr(UnityEnvWrapper, 'reset')
    assert hasattr(UnityEnvWrapper, 'step')
    assert hasattr(UnityEnvWrapper, 'close')
    assert hasattr(UnityEnvWrapper, 'n_agents')
    assert hasattr(UnityEnvWrapper, 'brain_names')
    assert hasattr(UnityEnvWrapper, 'observation_spaces')
    assert hasattr(UnityEnvWrapper, 'action_spaces')


@pytest.mark.skipif(UnityEnvWrapper is None, reason="UnityEnvWrapper not available")
def test_wrapper_init_signature():
    """Test that UnityEnvWrapper.__init__ has the expected signature."""
    import inspect
    sig = inspect.signature(UnityEnvWrapper.__init__)
    params = list(sig.parameters.keys())

    # Check that it has the expected parameters
    assert 'self' in params
    assert 'env_id' in params
    assert 'worker_id' in params
    assert 'seed' in params
    assert 'no_graphics' in params
