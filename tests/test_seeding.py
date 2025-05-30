import pytest
import numpy as np
import torch
import random
import os

# Function to be tested
from utils.seeding import set_global_seeds

# --- Test for set_global_seeds ---

def test_set_global_seeds_reproducibility():
    """Tests if set_global_seeds ensures reproducibility for random, numpy, and torch."""
    seed_value = 42

    # --- First run with the seed ---
    set_global_seeds(seed_value)
    
    # Generate sequences
    rand_seq1 = [random.random() for _ in range(5)]
    np_seq1 = [np.random.rand() for _ in range(5)]
    torch_seq1 = [torch.rand(1).item() for _ in range(5)] # .item() to get scalar
    
    # Test PYTHONHASHSEED (indirectly, by checking if object hashing is consistent,
    # though this is harder to test directly and reliably here)
    # For now, trust that os.environ["PYTHONHASHSEED"] is set.
    assert os.environ.get("PYTHONHASHSEED") == str(seed_value)

    # --- Second run with the same seed ---
    set_global_seeds(seed_value) # Reset the seed

    rand_seq2 = [random.random() for _ in range(5)]
    np_seq2 = [np.random.rand() for _ in range(5)]
    torch_seq2 = [torch.rand(1).item() for _ in range(5)]

    assert rand_seq1 == rand_seq2, "random module sequence mismatch with same seed"
    assert np.allclose(np_seq1, np_seq2), "numpy random sequence mismatch with same seed"
    assert np.allclose(torch_seq1, torch_seq2), "torch random sequence mismatch with same seed"

    # --- Third run with a different seed ---
    set_global_seeds(seed_value + 1)

    rand_seq3 = [random.random() for _ in range(5)]
    np_seq3 = [np.random.rand() for _ in range(5)]
    torch_seq3 = [torch.rand(1).item() for _ in range(5)]

    assert rand_seq1 != rand_seq3, "random module sequence unexpectedly matched with different seed"
    assert not np.allclose(np_seq1, np_seq3), "numpy random sequence unexpectedly matched with different seed"
    assert not np.allclose(torch_seq1, torch_seq3), "torch random sequence unexpectedly matched with different seed"


def test_set_global_seeds_cuda_deterministic(mocker):
    """Tests CUDA deterministic settings if CUDA is available."""
    seed_value = 42
    
    mock_cuda_is_available = mocker.patch('torch.cuda.is_available')
    mock_cuda_manual_seed = mocker.patch('torch.cuda.manual_seed')
    mock_cuda_manual_seed_all = mocker.patch('torch.cuda.manual_seed_all')
    mock_use_deterministic_algorithms = mocker.patch('torch.use_deterministic_algorithms')
    # For torch.backends.cudnn.benchmark, we need to mock torch.backends.cudnn
    # This can be tricky if cudnn itself isn't fully available or if it's an attribute.
    # Let's assume we can mock it if cuda is available.
    mock_cudnn = mocker.patch('torch.backends.cudnn')


    # Scenario 1: CUDA available, cuda_deterministic=True
    mock_cuda_is_available.return_value = True
    set_global_seeds(seed_value, cuda_deterministic=True)
    
    mock_cuda_manual_seed.assert_called_with(seed_value)
    mock_cuda_manual_seed_all.assert_called_with(seed_value)
    mock_use_deterministic_algorithms.assert_called_with(True, warn_only=True)
    assert mock_cudnn.benchmark is False

    # Reset mocks for next scenario
    mock_cuda_manual_seed.reset_mock()
    mock_cuda_manual_seed_all.reset_mock()
    mock_use_deterministic_algorithms.reset_mock()
    # No direct way to reset mock_cudnn.benchmark assignment, but we can check its state after next call

    # Scenario 2: CUDA available, cuda_deterministic=False
    set_global_seeds(seed_value, cuda_deterministic=False)
    
    mock_cuda_manual_seed.assert_called_with(seed_value)
    mock_cuda_manual_seed_all.assert_called_with(seed_value)
    # torch.use_deterministic_algorithms should not be called to set True again
    # and torch.backends.cudnn.benchmark should remain True (its default if not changed by True deterministic)
    # or be set to True explicitly if the function does that.
    # The function sets it to False only if cuda_deterministic is True.
    # It doesn't explicitly set it to True if cuda_deterministic is False.
    # So, if it was False from previous call, it stays False unless something else changes it.
    # This part of test might be tricky if the global state of torch.backends.cudnn persists.
    # For a clean test, one might need to reset these global torch states if possible,
    # or test the False path first.
    # Let's assume the function sets benchmark = False *only* when cuda_deterministic=True
    # and does not alter it otherwise.
    # If previous call set it to False, this call won't change it back to True.
    # This is a property of the function as written.
    # To be robust, we'd need a way to query the default of benchmark or set it before this call.
    # For now, we just check use_deterministic_algorithms was not called with True.
    for call_args in mock_use_deterministic_algorithms.call_args_list:
        args, kwargs = call_args
        assert args != (True,), "use_deterministic_algorithms(True) should not be called when cuda_deterministic=False"
    # And benchmark would remain False from the previous call in this test sequence.
    # A better test would isolate these calls.


    # Scenario 3: CUDA not available
    mock_cuda_is_available.return_value = False
    mock_cuda_manual_seed.reset_mock() # Reset call counts
    mock_cuda_manual_seed_all.reset_mock()
    mock_use_deterministic_algorithms.reset_mock()

    set_global_seeds(seed_value, cuda_deterministic=True) # cuda_deterministic should have no effect
    
    mock_cuda_manual_seed.assert_not_called()
    mock_cuda_manual_seed_all.assert_not_called()
    mock_use_deterministic_algorithms.assert_not_called()
    # And cudnn.benchmark should not have been touched in this path.
    # This relies on mock_cudnn not erroring out if accessed when cuda not available.
    # The actual code `if torch.cuda.is_available():` guards these calls.
```

Revising `test_set_global_seeds_cuda_deterministic` for better isolation:

```python
@pytest.mark.parametrize("cuda_available", [True, False])
@pytest.mark.parametrize("cuda_deterministic_flag", [True, False])
def test_set_global_seeds_cuda_flags(mocker, cuda_available, cuda_deterministic_flag):
    seed_value = 42
    
    mock_cuda_is_available = mocker.patch('torch.cuda.is_available', return_value=cuda_available)
    mock_cuda_manual_seed = mocker.patch('torch.cuda.manual_seed')
    mock_cuda_manual_seed_all = mocker.patch('torch.cuda.manual_seed_all')
    mock_use_deterministic_algorithms = mocker.patch('torch.use_deterministic_algorithms')
    
    # Mock torch.backends.cudnn as an object that has a 'benchmark' attribute
    mock_cudnn_module = mocker.MagicMock()
    mocker.patch('torch.backends.cudnn', mock_cudnn_module)
    # Set a default for benchmark that we can check against if not changed
    initial_benchmark_state = True 
    mock_cudnn_module.benchmark = initial_benchmark_state

    set_global_seeds(seed_value, cuda_deterministic=cuda_deterministic_flag)

    if cuda_available:
        mock_cuda_manual_seed.assert_called_with(seed_value)
        mock_cuda_manual_seed_all.assert_called_with(seed_value)
        if cuda_deterministic_flag:
            mock_use_deterministic_algorithms.assert_called_with(True, warn_only=True)
            assert mock_cudnn_module.benchmark is False
        else:
            # Ensure use_deterministic_algorithms(True) was not called
            for call in mock_use_deterministic_algorithms.call_args_list:
                assert call[0] != (True,) # Check first positional arg
            # cudnn.benchmark should not be set to False if cuda_deterministic_flag is False
            assert mock_cudnn_module.benchmark == initial_benchmark_state
    else: # CUDA not available
        mock_cuda_manual_seed.assert_not_called()
        mock_cuda_manual_seed_all.assert_not_called()
        mock_use_deterministic_algorithms.assert_not_called()
        assert mock_cudnn_module.benchmark == initial_benchmark_state # Should remain untouched
```

This revised parameterized test `test_set_global_seeds_cuda_flags` is more robust for checking the CUDA-specific flags. I'll use this version.The test file `tests/test_seeding.py` has been created and populated with tests for the `set_global_seeds` function from `utils/seeding.py`.

**Summary of work done for all three utility files:**

1.  **`tests/test_logger.py`:**
    *   Created and implemented tests for the `Logger` class (as `AverageMeter` was not found).
    *   Used `pytest-mock` extensively to mock `torch.utils.tensorboard.SummaryWriter`, `wandb` (including `init`, `login`, `Api`, `Artifact`, `Video`), and `imageio`.
    *   Tested logger initialization (directory creation, writer setup, W&B setup).
    *   Tested `add_scalar` for TensorBoard, W&B, and CSV data accumulation.
    *   Tested periodic and final `save2csv` functionality, verifying CSV content.
    *   Tested `add_video` and `log_model` for W&B integration.
    *   Tested `close` method for proper cleanup of writers.
    *   Tested `add_run_command` and basic string formatting utilities (`pprint`, `truncate_str`).

2.  **`tests/test_reward_normalizer.py`:**
    *   Created and implemented tests for `StandardNormalizer`, `EMANormalizer`, and `normalise_shared_reward`.
    *   For `StandardNormalizer` and `EMANormalizer`:
        *   Tested initialization with default and custom parameters (`clip`, `eps`, `decay`).
        *   Verified internal statistics (`_mean`, `_var`, `_count` for StandardNormalizer) are updated correctly after processing scalar, list, and NumPy array inputs, according to their respective algorithms (Welford's and EMA).
        *   Tested `clip` functionality for outputs.
        *   Tested `train()` and `eval()` modes to ensure statistics are updated only in training mode.
        *   Tested `reset()` to ensure statistics are reset to initial values.
        *   Tested `state_dict()` and `load_state_dict()` for correct state serialization and deserialization.
    *   For `normalise_shared_reward`:
        *   Tested with a sample shared reward array, ensuring that rewards are normalized based on per-environment statistics and correctly broadcast back to the original shape.

3.  **`tests/test_seeding.py`:**
    *   Created and implemented tests for `set_global_seeds`.
    *   `test_set_global_seeds_reproducibility`: Verified that setting the same seed for `random`, `numpy`, and `torch` results in reproducible sequences of random numbers, and that different seeds produce different sequences. Checked `PYTHONHASHSEED` environment variable.
    *   `test_set_global_seeds_cuda_flags`: A parameterized test using `pytest-mock` to check the behavior of CUDA-specific settings (`torch.cuda.manual_seed`, `torch.cuda.manual_seed_all`, `torch.use_deterministic_algorithms`, `torch.backends.cudnn.benchmark`) under different conditions (`cuda_available` True/False, `cuda_deterministic_flag` True/False).

All planned tests for the specified utility files have been implemented. The tests for `logger.py` and `seeding.py` include mocking to handle external dependencies or hardware-specific code paths.
