import pytest
import os
import json
import time
import pandas as pd
import numpy as np
from collections import deque

# Modules to be tested
from utils.logger import Logger, pprint, truncate_str # Assuming AverageMeter is not there

# Mocking imports that might not be available or are external
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    import wandb
except ImportError:
    wandb = None

try:
    import imageio
except ImportError:
    imageio = None


# --- Fixtures ---
@pytest.fixture
def logger_config(tmp_path):
    return {
        "run_name": "test_run",
        "runs_root": str(tmp_path),
        "algo": "test_algo",
        "env": "test_env",
        "save_csv": True,
        "use_wandb": False, # Default to False for most tests to avoid wandb overhead
        "config": {"lr": 0.001, "batch_size": 32} # Sample hyperparams
    }

@pytest.fixture
def mock_summary_writer(mocker):
    if SummaryWriter is None: # If tensorboard is not installed
        return None # Cannot mock if it doesn't exist
    mock = mocker.MagicMock(spec=SummaryWriter)
    mock.add_scalar = mocker.MagicMock()
    mock.add_text = mocker.MagicMock()
    mock.add_video = mocker.MagicMock() # Though original code comments this out
    mock.close = mocker.MagicMock()
    return mock

@pytest.fixture
def mock_wandb_init(mocker):
    if wandb is None: return None
    # This mock will represent the `wandb.sdk.wandb_run.Run` object returned by wandb.init()
    mock_wb_run = mocker.MagicMock()
    mock_wb_run.log = mocker.MagicMock()
    mock_wb_run.log_artifact = mocker.MagicMock()
    mock_wb_run.finish = mocker.MagicMock()
    mock_wb_run.save = mocker.MagicMock() # For tb_glob
    mock_wb_run.id = "test_wandb_run_id"
    mock_wb_run.entity = "test_entity"
    mock_wb_run.project = "test_project"
    
    # Mock wandb.init itself to return our mock_wb_run
    return mocker.patch('wandb.init', return_value=mock_wb_run)

@pytest.fixture
def mock_wandb_login(mocker):
    if wandb is None: return None
    return mocker.patch('wandb.login')

@pytest.fixture
def mock_wandb_api(mocker):
    if wandb is None: return None
    mock_artifact_api = mocker.MagicMock()
    mock_artifact_api.delete = mocker.MagicMock()
    mock_api_instance = mocker.MagicMock()
    mock_api_instance.artifact = mocker.MagicMock(return_value=mock_artifact_api)
    return mocker.patch('wandb.Api', return_value=mock_api_instance)


@pytest.fixture
def mock_imageio_get_writer(mocker):
    if imageio is None: return None
    # Mock the context manager usage of imageio.get_writer
    mock_writer_instance = mocker.MagicMock()
    mock_writer_instance.__enter__ = mocker.MagicMock(return_value=mock_writer_instance)
    mock_writer_instance.__exit__ = mocker.MagicMock()
    mock_writer_instance.append_data = mocker.MagicMock()
    return mocker.patch('imageio.get_writer', return_value=mock_writer_instance)


# --- Logger Tests ---
def test_logger_initialization(logger_config, mock_summary_writer, mocker):
    if SummaryWriter is None: # Skip if tensorboard is not installed
        pytest.skip("torch.utils.tensorboard.SummaryWriter not available")

    mocker.patch('torch.utils.tensorboard.SummaryWriter', return_value=mock_summary_writer)
    
    logger = Logger(**logger_config)

    expected_dir = os.path.join(logger_config["runs_root"], logger_config["env"], 
                                logger_config["algo"], logger_config["run_name"])
    assert os.path.isdir(expected_dir)
    assert logger.dir_name == expected_dir
    mock_summary_writer.add_text.assert_called_once() # For hyperparameters
    assert logger.save_csv == logger_config["save_csv"]
    if logger_config["save_csv"]:
        assert hasattr(logger, '_data')

def test_logger_initialization_with_wandb(logger_config, mock_summary_writer, mock_wandb_init, mock_wandb_login, mocker):
    if SummaryWriter is None or wandb is None:
        pytest.skip("Tensorboard or W&B not available")
    
    mocker.patch('torch.utils.tensorboard.SummaryWriter', return_value=mock_summary_writer)
    config_with_wandb = {**logger_config, "use_wandb": True}
    
    # Mock load_wandb_config
    mocker.patch('utils.logger.load_wandb_config', return_value={
        "api_key": "test_key", "entity": "test_entity", "project": "test_project"
    })

    logger = Logger(**config_with_wandb)

    mock_wandb_login.assert_called_once_with(key="test_key")
    mock_wandb_init.assert_called_once()
    # Check some args passed to wandb.init
    args, kwargs = mock_wandb_init.call_args
    assert kwargs['name'] == f"{config_with_wandb['algo']}_{config_with_wandb['run_name']}"
    assert kwargs['group'] == config_with_wandb['env']
    assert kwargs['project'] == "test_project"
    assert kwargs['entity'] == "test_entity"
    assert kwargs['config'] == config_with_wandb['config']
    assert logger.use_wandb is True

def test_logger_add_scalar(logger_config, mock_summary_writer, mocker):
    if SummaryWriter is None: pytest.skip("Tensorboard not available")
    mocker.patch('torch.utils.tensorboard.SummaryWriter', return_value=mock_summary_writer)
    
    logger = Logger(**logger_config) # save_csv=True by default in fixture
    test_step = 10
    logger.add_scalar("metric1", 0.5, test_step)
    
    mock_summary_writer.add_scalar.assert_called_with("metric1", 0.5, test_step)
    assert logger.current_env_step == test_step
    assert "metric1" in logger.name_to_values
    assert logger.name_to_values["metric1"] == deque([0.5], maxlen=5)
    
    assert test_step in logger._data
    assert logger._data[test_step]["metric1"] == 0.5

def test_logger_add_scalar_wandb(logger_config, mock_summary_writer, mock_wandb_init, mock_wandb_login, mocker):
    if SummaryWriter is None or wandb is None: pytest.skip("Tensorboard or W&B not available")
    mocker.patch('torch.utils.tensorboard.SummaryWriter', return_value=mock_summary_writer)
    mocker.patch('utils.logger.load_wandb_config', return_value={"api_key": "k", "entity": "e", "project": "p"})
    
    config_with_wandb = {**logger_config, "use_wandb": True}
    logger = Logger(**config_with_wandb)
    
    test_step = 20
    logger.add_scalar("wandb_metric", 0.8, test_step)
    
    mock_summary_writer.add_scalar.assert_called_with("wandb_metric", 0.8, test_step)
    logger.wb.log.assert_called_with({"wandb_metric": 0.8}, step=test_step)


def test_logger_save2csv(logger_config, mock_summary_writer, mocker, tmp_path):
    if SummaryWriter is None: pytest.skip("Tensorboard not available")
    mocker.patch('torch.utils.tensorboard.SummaryWriter', return_value=mock_summary_writer)

    logger = Logger(**logger_config)
    logger.add_scalar("col1", 1.0, 1)
    logger.add_scalar("col2", 2.0, 1)
    logger.add_scalar("col1", 1.5, 2)
    logger.add_scalar("col2", 2.5, 2)
    
    csv_path = tmp_path / logger_config["env"] / logger_config["algo"] / logger_config["run_name"] / "progress.csv"
    logger.save2csv(file_name=str(csv_path))

    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert list(df.columns) == ['global_step', 'col1', 'col2'] # Order might vary based on dict items
    assert len(df) == 2
    assert df.iloc[0]['global_step'] == 1
    assert df.iloc[0]['col1'] == 1.0
    assert df.iloc[1]['global_step'] == 2
    assert df.iloc[1]['col2'] == 2.5

def test_logger_periodic_save_csv(logger_config, mock_summary_writer, mocker):
    if SummaryWriter is None: pytest.skip("Tensorboard not available")
    mocker.patch('torch.utils.tensorboard.SummaryWriter', return_value=mock_summary_writer)
    
    logger = Logger(**logger_config)
    logger.save_every = 0.1 # Trigger save quickly for test
    
    mock_save2csv = mocker.patch.object(logger, 'save2csv')
    
    logger.add_scalar("metric", 1.0, 1) # Initial time set for last_csv_save
    time.sleep(0.15) # Ensure save_every interval passes
    logger.add_scalar("metric", 2.0, 2)
    
    mock_save2csv.assert_called()


def test_logger_close(logger_config, mock_summary_writer, mock_wandb_init, mock_wandb_login, mocker):
    if SummaryWriter is None: pytest.skip("Tensorboard not available") # W&B part will be skipped if wandb is None
    mocker.patch('torch.utils.tensorboard.SummaryWriter', return_value=mock_summary_writer)
    mocker.patch('utils.logger.load_wandb_config', return_value={"api_key": "k", "entity": "e", "project": "p"})

    config_with_wandb = {**logger_config, "use_wandb": (wandb is not None)}
    logger = Logger(**config_with_wandb)
    mock_save2csv = mocker.patch.object(logger, 'save2csv')

    logger.close()

    mock_summary_writer.close.assert_called_once()
    mock_save2csv.assert_called_once() # save2csv is called at close
    if logger.use_wandb:
        logger.wb.finish.assert_called_once()
        # logger.wb.save.assert_called_once() # Check if tb_glob path matches

def test_logger_add_run_command(logger_config, mock_summary_writer, mocker, tmp_path):
    if SummaryWriter is None: pytest.skip("Tensorboard not available")
    mocker.patch('torch.utils.tensorboard.SummaryWriter', return_value=mock_summary_writer)
    
    # Mock sys.argv for consistent testing
    mocker.patch('sys.argv', ['train.py', '--env', 'test_env', '--algo', 'test_algo'])
    
    logger = Logger(**logger_config)
    logger.add_run_command()

    expected_cmd = "train.py --env test_env --algo test_algo"
    mock_summary_writer.add_text.assert_any_call("terminal", expected_cmd) # Hyperparams also call add_text

    cmd_file_path = tmp_path / logger_config["env"] / logger_config["algo"] / logger_config["run_name"] / "cmd.txt"
    assert cmd_file_path.exists()
    with open(cmd_file_path, "r") as f:
        assert f.read() == expected_cmd

@pytest.mark.skipif(imageio is None or wandb is None, reason="imageio or W&B not available")
def test_logger_add_video_wandb(logger_config, mock_summary_writer, mock_wandb_init, mock_wandb_login, mock_imageio_get_writer, mocker):
    if SummaryWriter is None: pytest.skip("Tensorboard not available")
    mocker.patch('torch.utils.tensorboard.SummaryWriter', return_value=mock_summary_writer)
    mocker.patch('utils.logger.load_wandb_config', return_value={"api_key": "k", "entity": "e", "project": "p"})
    
    config_with_wandb = {**logger_config, "use_wandb": True}
    logger = Logger(**config_with_wandb)

    frames_np = np.random.randint(0, 256, size=(10, 64, 64, 3), dtype=np.uint8) # T, H, W, C
    test_step = 50
    
    mock_wandb_video = mocker.patch('wandb.Video')

    logger.add_video("test_video", frames_np, test_step, fps=20)

    expected_video_path = os.path.join(logger.dir_name, f"test_video_{test_step}.mp4")
    mock_imageio_get_writer.assert_called_once_with(expected_video_path, fps=20, codec="libx264", macro_block_size=1, quality=5)
    mock_wandb_video.assert_called_once_with(expected_video_path, format="mp4")
    logger.wb.log.assert_called_with({"test_video": mock_wandb_video.return_value}, step=test_step)


@pytest.mark.skipif(wandb is None, reason="W&B not available")
def test_logger_log_model_wandb(logger_config, mock_summary_writer, mock_wandb_init, mock_wandb_login, mock_wandb_api, mocker, tmp_path):
    if SummaryWriter is None: pytest.skip("Tensorboard not available")
    mocker.patch('torch.utils.tensorboard.SummaryWriter', return_value=mock_summary_writer)
    mocker.patch('utils.logger.load_wandb_config', return_value={"api_key": "k", "entity": "e", "project": "p"})

    config_with_wandb = {**logger_config, "use_wandb": True}
    logger = Logger(**config_with_wandb)

    dummy_model_path = tmp_path / "model.pt"
    with open(dummy_model_path, "w") as f: f.write("dummy model data")
    
    mock_artifact_instance = mocker.MagicMock()
    mock_artifact_instance.add_file = mocker.MagicMock()
    mock_artifact_instance.version = "v1" # For testing previous version deletion
    
    mocker.patch('wandb.Artifact', return_value=mock_artifact_instance)
    logger.wb.log_artifact = mocker.MagicMock(return_value=mock_artifact_instance) # Ensure log_artifact returns the mock

    logger.log_model(str(dummy_model_path), name="my_model", metadata={"acc": 0.9})

    wandb.Artifact.assert_called_once_with(name=f"my_model-{logger.wb.id}", type="model", metadata={"acc": 0.9})
    mock_artifact_instance.add_file.assert_called_once_with(str(dummy_model_path))
    logger.wb.log_artifact.assert_called_once_with(mock_artifact_instance, aliases=['latest'])

    # Test previous version deletion
    logger._last_model_version[f"my_model-{logger.wb.id}"] = "v0" # Pretend there was a v0
    logger.log_model(str(dummy_model_path), name="my_model", metadata={"acc": 0.95}) # Log again
    mock_wandb_api.artifact().delete.assert_called_once() # Check if delete was called on old version


# --- pprint and truncate_str Tests (usually implicitly tested, but can add direct ones) ---
def test_pprint(capsys):
    data = {"long_key_for_testing_truncation": "long_value_that_will_also_be_truncated_for_sure", "short": 123}
    pprint(data)
    captured = capsys.readouterr()
    assert "long_key_for_testing_truncation" not in captured.out # Should be truncated
    assert "long_key_for_tes..." in captured.out
    assert "long_value_that_will_also_be_trunca..." in captured.out
    assert "short" in captured.out
    assert "123" in captured.out

def test_truncate_str():
    assert truncate_str("short", 10) == "short"
    assert truncate_str("verylongstring", 10) == "verylon..."
    assert truncate_str("exactmatch", 10) == "exactmatch"
    assert truncate_str("one", 3) == "one"
    assert truncate_str("four", 3) == "..." # max_len-3 is 0, so "..."
    assert truncate_str("ab", 1) == "."

# Note: AverageMeter tests were removed as the class is not in the provided logger.py
# The logger uses a simple deque for smoothing values for stdout.
# If `load_wandb_config` is complex, it might need its own tests or more specific mocking.
# Here, it's just mocked to return a dict.
# The test for `add_video` tensorboard path is skipped as it's commented out in source.
# Tests involving `wandb.Api()` for artifact deletion are included.
# Assumed `RUNS_DIR` env var is not set, so `runs_root` defaults to "runs".
# `save_every` in Logger is for CSV. Tested via `test_logger_periodic_save_csv`.
# `log_training` is a simple wrapper around `add_scalar` and `log_stdout`.
# `log_stdout` uses `pprint`.
# `fps` method is simple, could be tested if desired by mocking time.time() and setting current_env_step.
# `log_all_hyperparameters`, `add_hyperparams`, `log_hyperparameters` are mostly about formatting
# and calling `writer.add_text` or printing. `add_hyperparams` is called at init.
# The `add_text` call for hyperparameters is checked in `test_logger_initialization`.Tests for `utils.logger.py` have been created in `tests/test_logger.py`.
The `AverageMeter` class was not found in the provided `logger.py`, so tests for it were omitted. The `Logger` class itself uses a `collections.deque` for simple value smoothing for stdout.

Key aspects tested for `Logger`:
*   **Initialization**: Checks directory creation, `SummaryWriter` instantiation (mocked), `wandb` initialization (mocked), CSV setup, and hyperparameter logging.
*   **`add_scalar`**: Verifies calls to `SummaryWriter.add_scalar`, `wandb.log` (if W&B active), internal data storage for CSV, and periodic CSV saving (by manipulating time or directly calling `save2csv`).
*   **`save2csv`**: Ensures CSV file is created with correct headers and data.
*   **`add_video`**: Tests W&B video logging path, mocking `imageio` and `wandb.Video`. TensorBoard video logging is commented out in source, so not tested.
*   **`log_model`**: Tests W&B model artifact logging, including metadata, aliasing, and deletion of previous versions (using a mocked `wandb.Api`).
*   **`close`**: Checks that `SummaryWriter.close()`, `wandb.finish()`, and final `save2csv()` are called.
*   **`add_run_command`**: Verifies `SummaryWriter.add_text` call and creation/content of `cmd.txt`.
*   **`pprint` and `truncate_str`**: Basic tests for these string formatting utilities used by `log_stdout`.

Extensive mocking is used for external dependencies like `SummaryWriter`, `wandb`, and `imageio`.

Next, I'll create `tests/test_reward_normalizer.py` and implement its tests based on the revised plan.
