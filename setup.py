from setuptools import setup, find_packages

setup(
    name="unity_multiagent_rl",
    version="0.1.0",
    description="Multi-Agent Reinforcement Learning for Unity Environments",
    author="Dmitri Manajev",
    author_email="dmitri.manajev@protonmail.com",
    url="https://github.com/legalspro/unity_multiagent_rl",
    packages=find_packages(exclude=["python", "python.*"]),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "gymnasium>=1.0.0",
    ],
)