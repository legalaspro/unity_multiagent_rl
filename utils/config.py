import os
from dotenv import load_dotenv

def load_wandb_config():
    """Load W&B configuration from environment variables"""
    load_dotenv(override=True)
    
    return {
        "api_key": os.getenv("WANDB_API_KEY"),
        "entity": os.getenv("WANDB_ENTITY"),
        "project": os.getenv("WANDB_PROJECT", "unity-multiagent-rl"),  # fallback to default
    }