import pytest
import argparse
import yaml
import os
from types import SimpleNamespace

# Functions to be tested (assuming they are in utils.arg_tools)
from utils.arg_tools import load_config, merge_cli, _explicit_cli_keys

# --- Fixtures ---
@pytest.fixture
def temp_config_files(tmp_path):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    algos_dir = configs_dir / "algos"
    algos_dir.mkdir()

    base_yaml_content = {"param_base": "base_value", "shared_param": "base_shared"}
    with open(configs_dir / "base.yaml", "w") as f:
        yaml.dump(base_yaml_content, f)

    algo_yaml_content = {"param_algo": "algo_value", "shared_param": "algo_shared"}
    with open(algos_dir / "test_algo.yaml", "w") as f:
        yaml.dump(algo_yaml_content, f)
    
    override_yaml_content = {"param_override": "override_value", "shared_param": "override_shared"}
    override_file = tmp_path / "override.yaml"
    with open(override_file, "w") as f:
        yaml.dump(override_yaml_content, f)
        
    # Return paths relative to tmp_path for load_config to work if it uses relative paths
    # However, load_config uses absolute paths like "configs/base.yaml".
    # So, we need to monkeypatch the open() call or change directory.
    # For simplicity, we'll assume load_config can be tested by ensuring these files
    # are in the expected relative locations from where pytest is run, or use monkeypatch.
    # Let's try chdir for this test.
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield {
        "base": "configs/base.yaml", # Path relative to tmp_path
        "algo": "configs/algos/test_algo.yaml",
        "override": "override.yaml" # Path relative to tmp_path
    }
    os.chdir(original_cwd)


# --- Tests for load_config ---
def test_load_config_base_and_algo(temp_config_files):
    # temp_config_files fixture changes cwd to tmp_path
    cfg = load_config(algo="test_algo", config_path=None)
    
    assert cfg["param_base"] == "base_value"
    assert cfg["param_algo"] == "algo_value"
    assert cfg["shared_param"] == "algo_shared" # Algo should override base

def test_load_config_with_override(temp_config_files):
    cfg = load_config(algo="test_algo", config_path=temp_config_files["override"])
    
    assert cfg["param_base"] == "base_value"
    assert cfg["param_algo"] == "algo_value"
    assert cfg["param_override"] == "override_value"
    assert cfg["shared_param"] == "override_shared" # Override should override algo and base

def test_load_config_missing_algo_file(temp_config_files):
    with pytest.raises(FileNotFoundError):
        load_config(algo="non_existent_algo", config_path=None)

# --- Tests for _explicit_cli_keys ---
def test_explicit_cli_keys():
    argv1 = ["--lr", "0.001", "some_pos_arg", "--bool_flag"]
    assert _explicit_cli_keys(argv1) == {"lr", "bool_flag"}

    argv2 = ["--path=some/path", "--another-key=value"]
    assert _explicit_cli_keys(argv2) == {"path", "another-key"}
    
    argv3 = ["--key"] # Flag without value immediately after
    assert _explicit_cli_keys(argv3) == {"key"}

    argv4 = []
    assert _explicit_cli_keys(argv4) == set()

# --- Tests for merge_cli ---
@pytest.fixture
def base_cfg_dict():
    return {"lr": 0.1, "gamma": 0.99, "arch": "cnn"}

def test_merge_cli_known_args_override(base_cfg_dict):
    # Known args explicitly set
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float)
    parser.add_argument("--new_param", type=str, default="default_new")
    
    cli_known_args = parser.parse_args(["--lr", "0.005"])
    argv = ["--lr", "0.005"] # lr is explicit
    
    merged_ns = merge_cli(base_cfg_dict.copy(), cli_known_args, [], argv=argv)
    
    assert isinstance(merged_ns, SimpleNamespace)
    assert merged_ns.lr == 0.005 # Overridden by explicit CLI
    assert merged_ns.gamma == 0.99 # From base_cfg_dict
    assert merged_ns.arch == "cnn" # From base_cfg_dict
    assert merged_ns.new_param == "default_new" # Default from parser, not in argv, not in base_cfg

def test_merge_cli_known_args_default_behavior(base_cfg_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01) # Default different from base_cfg
    parser.add_argument("--arch", type=str, default="mlp") # Default different from base_cfg
    parser.add_argument("--another_param", type=int, default=100) # Not in base_cfg

    # Scenario 1: --lr is NOT in argv (i.e., user relies on its default)
    # but 'lr' IS in base_cfg_dict. merge_cli should keep base_cfg_dict's value.
    cli_known_args_1 = parser.parse_args([]) # No explicit args
    argv_1 = [] 
    merged_ns_1 = merge_cli(base_cfg_dict.copy(), cli_known_args_1, [], argv=argv_1)
    assert merged_ns_1.lr == 0.1 # Kept from base_cfg_dict, not parser default
    assert merged_ns_1.arch == "cnn" # Kept from base_cfg_dict

    # Scenario 2: --another_param is NOT in argv, and 'another_param' is NOT in base_cfg_dict.
    # merge_cli should add it from parser defaults.
    assert merged_ns_1.another_param == 100 # Added from parser default

def test_merge_cli_unknown_args_override(base_cfg_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    cli_known_args = parser.parse_args(["--lr", "0.005"]) # lr is explicit here
    
    argv = ["--lr", "0.005"] # lr is explicit
    # Unknown args should take highest precedence
    unknown_cli = ["--lr", "1.0e-5", "--gamma", "0.95", "--new_unknown", "true"]
    
    merged_ns = merge_cli(base_cfg_dict.copy(), cli_known_args, unknown_cli, argv=argv)
    
    assert merged_ns.lr == 1e-5 # Overridden by unknown_cli
    assert merged_ns.gamma == 0.95 # Overridden by unknown_cli
    assert merged_ns.arch == "cnn" # From base_cfg_dict
    assert merged_ns.new_unknown is True # Added by unknown_cli, type converted by yaml.safe_load

def test_merge_cli_type_conversion_for_unknown(base_cfg_dict):
    cli_known_args = argparse.Namespace() # No known args for simplicity
    argv = []
    unknown_cli = ["--episodes", "1000", "--rate", "0.0025", "--active", "false", "--layers", "[64, 32]"]
    
    merged_ns = merge_cli(base_cfg_dict.copy(), cli_known_args, unknown_cli, argv=argv)
    
    assert merged_ns.episodes == 1000
    assert isinstance(merged_ns.episodes, int)
    assert merged_ns.rate == 0.0025
    assert isinstance(merged_ns.rate, float)
    assert merged_ns.active is False
    assert isinstance(merged_ns.active, bool)
    assert merged_ns.layers == [64, 32]
    assert isinstance(merged_ns.layers, list)

def test_merge_cli_empty_inputs(base_cfg_dict):
    cli_known_args = argparse.Namespace()
    argv = []
    unknown_cli = []
    merged_ns = merge_cli(base_cfg_dict.copy(), cli_known_args, unknown_cli, argv=argv)
    
    assert merged_ns.lr == 0.1
    assert merged_ns.gamma == 0.99
    assert merged_ns.arch == "cnn"
    # Check no other attrs were added
    assert len(vars(merged_ns)) == 3

# It seems `dict_to_namespace` is not directly exposed, but `merge_cli` returns `SimpleNamespace`
# which achieves a similar goal. If `dict_to_namespace` was a standalone utility, its test would be:
# def test_dict_to_namespace():
#     d = {"a": 1, "b": {"c": 2}}
#     ns = dict_to_namespace(d) # Assuming this function exists
#     assert isinstance(ns, SimpleNamespace)
#     assert ns.a == 1
#     assert isinstance(ns.b, SimpleNamespace)
#     assert ns.b.c == 2
# Since it's not, the test for merge_cli returning SimpleNamespace covers this implicitly.
# The functions `get_config_from_yaml` and `get_args_from_yaml` are not in the provided code.
# `load_config` is the equivalent of `get_config_from_yaml`.
# `argparse.ArgumentParser.parse_known_args()` along with `merge_cli` handles CLI and config merging.
# The overall process of loading YAML and then merging with CLI to get a namespace is covered.
