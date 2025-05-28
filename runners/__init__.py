"""
Runner scripts for training and evaluating agents.
"""
from .off_policy_runner import OffPolicyRunner
from .on_policy_runner import OnPolicyRunner
from .render_runner import RenderRunner

RUNNER_REGISTRY = {
    'maddpg': OffPolicyRunner,
    'mappo': OnPolicyRunner,
    'masac': OffPolicyRunner,
    'matd3': OffPolicyRunner
}

__all__ = [
    'OffPolicyRunner',
    'OnPolicyRunner',
    'RenderRunner'
]