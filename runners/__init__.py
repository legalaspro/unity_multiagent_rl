"""
Runner scripts for training and evaluating agents.
"""
from .off_policy_runner import OffPolicyRunner
from .on_policy_runner import OnPolicyRunner

RUNNER_REGISTRY = {
    'maddpg': OffPolicyRunner,
    'mappo': OnPolicyRunner,
    'masac': OffPolicyRunner,
    'matd3': OffPolicyRunner
}