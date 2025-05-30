# Import only what's needed for tests to avoid TensorFlow dependency issues
# The unitytrainers module imports TensorFlow which is not available in CI
try:
    from unityagents import *
except ImportError:
    pass

# Don't import unitytrainers to avoid TensorFlow dependency
# from unitytrainers import *
