import argparse
import sys
from types import SimpleNamespace
from typing import Optional

import yaml

def load_config(algo: str, config_path: Optional[str]):
    """
    Load config from YAML files.
    """
    cfg = {}

    # base defaults
    with open("configs/base.yaml") as f:
        cfg.update(yaml.safe_load(f))

    # algorithm defaults
    with open(f"configs/algos/{algo}.yaml") as f:
        cfg.update(yaml.safe_load(f))

    # config overrides
    if config_path:
        with open(config_path) as f:
            cfg.update(yaml.safe_load(f))

    return cfg

def merge_cli(
        cfg: dict, 
        cli: argparse.Namespace, 
        unknown_cli: list[str],
        argv: list[str] | None = None):
    """
    Merge precedence (lowest → highest):
      1. cfg dict   (already contains YAML values)
      2. explicit *known* CLI flags
      3. key/value pairs given in `unknown_cli`
    """
    if argv is None:                    # allows easier unit-testing
        argv = sys.argv[1:]

    explicit = _explicit_cli_keys(argv)

    # known flags first
    for k, v in vars(cli).items():
        if k in explicit or k not in cfg:
            cfg[k] = v

    # unknown flags come as ["--lr_actor", "3e-4", "--buffer_size", "1e6"]
    key = None
    for tok in unknown_cli:
        if tok.startswith("--"):
            key = tok.lstrip("-")
        else:
            cfg[key] = yaml.safe_load(tok)
    return SimpleNamespace(**cfg)

def _explicit_cli_keys(argv: list[str]) -> set[str]:
    """
    Return the set of `--flag` names that **actually appeared**
    on the command line (ignores values).
    """
    keys = set()
    for tok in argv:
        if tok.startswith("--"):
            key = tok.lstrip("-")
            # strip any trailing "=value" (handled by `prog --lr=3e-4`)
            key = key.split("=", 1)[0]
            keys.add(key)
    return keys
